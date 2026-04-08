import os
import logging
import asyncio
import json
from datetime import datetime
from collections import Counter

import psycopg2
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from google import genai

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === DB CONNECTION ===
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

# === INIT DB ===
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        user_id BIGINT,
        user_name TEXT,
        chat_id BIGINT,
        text TEXT,
        embedding TEXT,
        topic TEXT,
        time TIMESTAMP
    )
    """)

    conn.commit()
    cur.close()
    conn.close()

init_db()

# === CACHE ===
BOT_USERNAME = None
user_last_message = {}

async def get_bot_username(context):
    global BOT_USERNAME
    if not BOT_USERNAME:
        bot = await context.bot.get_me()
        BOT_USERNAME = bot.username.lower()
    return BOT_USERNAME

# === SPAM PROTECTION ===
def is_spamming(user_id):
    now = datetime.now().timestamp()
    if user_id in user_last_message and now - user_last_message[user_id] < 2:
        return True
    user_last_message[user_id] = now
    return False

# === EMBEDDING ===
async def get_embedding(text):
    try:
        response = await asyncio.to_thread(
            client.models.embed_content,
            model="embedding-001",
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

# === COSINE SIMILARITY ===
def cosine_similarity(a, b):
    if not a or not b:
        return 0
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)

# === TOPIC DETECTION (Burmese-friendly) ===
async def detect_topic(text):
    prompt = f"""
အောက်ပါစာကို အကြောင်းအရာအတိုချုံး (topic) ၁-၂ လုံးဖြင့် ပြောပါ။

Text:
{text}
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip().lower()
    except Exception as e:
        logger.error(f"Topic error: {e}")
        return "general"

# === SAVE MESSAGE ===
async def save_message(update: Update):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()

    embedding = await get_embedding(text)
    topic = await detect_topic(text)

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO messages (user_id, user_name, chat_id, text, embedding, topic, time)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            update.message.from_user.id,
            update.message.from_user.first_name,
            update.message.chat.id,
            text,
            json.dumps(embedding),  # SAFE storage
            topic,
            datetime.now()
        ))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"DB SAVE ERROR: {e}")

# === SEMANTIC SEARCH ===
async def get_relevant_messages(user_input, limit=10):
    query_embedding = await get_embedding(user_input)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT user_name, text, embedding FROM messages ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    scored = []

    for u, t, emb in rows:
        try:
            emb_list = json.loads(emb)
            score = cosine_similarity(query_embedding, emb_list)
            scored.append((score, u, t))
        except:
            continue

    scored.sort(reverse=True)
    return [(u, t) for score, u, t in scored[:limit] if score > 0.5]

# === TOPIC THREAD ===
async def get_topic_messages(user_input):
    topic = await detect_topic(user_input)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT user_name, text FROM messages
    WHERE topic = %s
    ORDER BY id DESC LIMIT 20
    """, (topic,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    rows.reverse()
    return rows

# === FORMAT ===
def format_messages(msgs):
    return "\n".join([f"{u}: {t}" for u, t in msgs])

# === AI RESPONSE ===
async def generate_ai(prompt):
    return await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

# === CHAT ===
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.message.from_user.id

    if is_spamming(user_id):
        return

    user_input = update.message.text
    bot_username = await get_bot_username(context)

    is_reply = (
        update.message.reply_to_message and
        update.message.reply_to_message.from_user.id == context.bot.id
    )

    if update.message.chat.type != "private":
        if not is_reply and f"@{bot_username}" not in user_input.lower():
            return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    # 🧠 Context building
    semantic_msgs = await get_relevant_messages(user_input)
    topic_msgs = await get_topic_messages(user_input)

    context_msgs = (semantic_msgs + topic_msgs)[-10:]

    prompt = f"""
You are a smart assistant.

IMPORTANT:
- Answer ONLY the user’s question
- Ignore unrelated context
- Always reply in Burmese

Context:
{format_messages(context_msgs)}

User:
{user_input}
"""

    try:
        response = await generate_ai(prompt)
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"AI ERROR: {e}")
        await update.message.reply_text("⚠️ AI မရရှိနိုင်ပါ။")

# === HANDLER ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_message(update)
    await chat(update, context)

# === APP ===
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

logger.info("Bot is running...")

app.run_polling(drop_pending_updates=True)