import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from collections import Counter

import psycopg2
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from google import genai
from google.genai import errors  # Added to catch specific API errors

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in Railway variables!")

client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === DATABASE FUNCTIONS ===
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

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

# === UTILS ===
BOT_USERNAME = None
user_last_message = {}

async def get_bot_username(context):
    global BOT_USERNAME
    if not BOT_USERNAME:
        bot = await context.bot.get_me()
        BOT_USERNAME = bot.username.lower()
    return BOT_USERNAME

def is_spamming(user_id):
    now = datetime.now().timestamp()
    if user_id in user_last_message and now - user_last_message[user_id] < 2:
        return True
    user_last_message[user_id] = now
    return False

# === AI PROCESSING ===
async def get_embedding(text):
    models_to_try = ["text-embedding-004", "embedding-001"]
    for model_name in models_to_try:
        try:
            response = await asyncio.to_thread(
                client.models.embed_content,
                model=model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.warning(f"Failed with {model_name}: {e}")
            continue
    return None

def cosine_similarity(a, b):
    if not a or not b: return 0
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)

async def detect_topic(text):
    prompt = f"Summarize this text into a short topic (1-2 words):\n\n{text}"
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip().lower()
    except:
        return "general"

# === DATABASE OPERATIONS ===
async def save_message(update: Update):
    if not update.message or not update.message.text: return
    text = update.message.text.strip()
    embedding = await get_embedding(text)
    topic = await detect_topic(text)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO messages (user_id, user_name, chat_id, text, embedding, topic, time)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (update.message.from_user.id, update.message.from_user.first_name,
          update.message.chat.id, text, json.dumps(embedding), topic, datetime.now()))
    conn.commit()
    cur.close()
    conn.close()

async def get_relevant_messages(user_input, limit=10):
    query_embedding = await get_embedding(user_input)
    if not query_embedding: return []
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_name, text, embedding FROM messages ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    scored = []
    for u, t, emb in rows:
        try:
            val = json.loads(emb)
            if val:
                score = cosine_similarity(query_embedding, val)
                scored.append((score, u, t))
        except: continue
    scored.sort(reverse=True)
    return [(u, t) for score, u, t in scored[:limit] if score > 0.5]

async def get_topic_messages(user_input):
    topic = await detect_topic(user_input)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_name, text FROM messages WHERE topic = %s ORDER BY id DESC LIMIT 20", (topic,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    rows.reverse()
    return rows

def format_messages(msgs):
    return "\n".join([f"{u}: {t}" for u, t in msgs])

# === AI CHAT HANDLER ===
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    if is_spamming(update.message.from_user.id): return

    user_input = update.message.text
    bot_username = await get_bot_username(context)
    is_reply = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id

    if update.message.chat.type != "private":
        if not is_reply and f"@{bot_username}" not in user_input.lower():
            return

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    semantic_msgs = await get_relevant_messages(user_input)
    topic_msgs = await get_topic_messages(user_input)
    context_msgs = (semantic_msgs + topic_msgs)[-10:]

    prompt = f"""
You are a very polite, warm, and helpful MALE AI assistant.
- In Burmese, use polite male particles: "ခင်ဗျာ" and "ခင်ဗျ".
- Tone: Caring gentleman.
- Emojis: 😊, 🙏, ✨.
- Language: Same as user.

Context:
{format_messages(context_msgs)}

User:
{user_input}
"""
    try:
        response = await asyncio.to_thread(client.models.generate_content, model="gemini-2.5-flash-lite", contents=prompt)
        await update.message.reply_text(response.text)
    except errors.ClientError as e:
        if e.code == 429:
            await update.message.reply_text("တောင်းပန်ပါတယ်ခင်ဗျာ။ ဒီနေ့အတွက် အခမဲ့အသုံးပြုခွင့် (Free Limit) ကုန်ဆုံးသွားလို့ပါခင်ဗျ။ ခဏနေမှဖြစ်ဖြစ်၊ မနက်ဖြန်မှဖြစ်ဖြစ် ပြန်ကြိုးစားပေးပါဦး ခင်ဗျာ။ 🙏")
        else:
            await update.message.reply_text("⚠️ စနစ်မှာ အမှားအယွင်းတစ်ခု ရှိနေလို့ပါ ခင်ဗျာ။")
    except:
        await update.message.reply_text("⚠️ စနစ်အနည်းငယ် အလုပ်များနေလို့ပါ ခင်ဗျာ။")

# === SUMMARY LOGIC ===
async def generate_summary(days):
    conn = get_conn()
    cur = conn.cursor()
    cutoff = datetime.now() - timedelta(days=days)
    cur.execute("SELECT user_name, text FROM messages WHERE time >= %s", (cutoff,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows: return "ပြောဆိုထားတဲ့ စာတိုလေးတွေ မတွေ့ရသေးပါဘူး ခင်ဗျာ။"

    text_data = "\n".join([f"{u}: {t}" for u, t in rows])
    prompt = f"Summarize this chat in Burmese with a polite male tone (ခင်ဗျာ/ခင်ဗျ):\n\n{text_data}"
    
    try:
        response = await asyncio.to_thread(client.models.generate_content, model="gemini-2.5-flash-lite", contents=prompt)
        return response.text
    except errors.ClientError as e:
        if e.code == 429:
            return "ဒီနေ့အတွက် အခမဲ့အသုံးပြုခွင့် (Free Limit) ကုန်ဆုံးသွားလို့ အကျဉ်းချုပ်ပေးလို့ မရသေးပါဘူး ခင်ဗျာ။ 🙏"
        return "⚠️ စနစ်မှာ အမှားအယွင်းတစ်ခု ဖြစ်နေပါတယ် ခင်ဗျာ။"
    except:
        return "⚠️ အကျဉ်းချုပ်ဖို့ အခက်အခဲရှိနေပါတယ် ခင်ဗျာ။"

# === COMMAND HANDLERS ===
async def todaysummary(update, context):
    await update.message.reply_text(await generate_summary(1))

async def yesterdaysummary(update, context):
    await update.message.reply_text(await generate_summary(2))

async def last3dayssummary(update, context):
    await update.message.reply_text(await generate_summary(3))

async def lastweeksummary(update, context):
    await update.message.reply_text(await generate_summary(7))

async def stats(update, context):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_name FROM messages")
    users = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    count = Counter(users)
    result = "📊 **အများဆုံး စကားပြောထားသူများစာရင်းပါ ခင်ဗျာ:**\n"
    for u, c in count.most_common(5):
        result += f"- {u}: {c} စာ\n"
    await update.message.reply_text(result)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_message(update)
    await chat(update, context)

# === START APPLICATION ===
if __name__ == "__main__":
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CommandHandler("todaysummary", todaysummary))
    app.add_handler(CommandHandler("yesterdaysummary", yesterdaysummary))
    app.add_handler(CommandHandler("last3dayssummary", last3dayssummary))
    app.add_handler(CommandHandler("lastweeksummary", lastweeksummary))
    app.add_handler(CommandHandler("stats", stats))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)