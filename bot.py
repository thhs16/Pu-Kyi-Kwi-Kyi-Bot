import os
import logging
import asyncio
from datetime import datetime, timedelta
from collections import Counter

import psycopg2
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from google import genai
from google.genai import errors

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === DATABASE ===
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

def is_why_question(text):
    text = text.lower()
    return any(word in text for word in [
        "why", "reason", "cause",
        "ဘာလို့", "ဘာကြောင့်", "အကြောင်း"
    ])

def filter_reason_messages(messages):
    keywords = ["because", "since", "so", "therefore", "due to",
                "လို့", "ကြောင့်", "အတွက်"]

    filtered = []
    for u, t, time in messages:
        if any(k in t.lower() for k in keywords):
            filtered.append((u, t, time))

    return filtered if filtered else messages

# === TOPIC ===
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

# === SAVE ===
async def save_message(update: Update):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    topic = await detect_topic(text)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO messages (user_id, user_name, chat_id, text, topic, time)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        update.message.from_user.id,
        update.message.from_user.first_name,
        update.message.chat.id,
        text,
        topic,
        datetime.now()
    ))

    conn.commit()
    cur.close()
    conn.close()

# === MEMORY ===
async def get_reply_context(update: Update):
    if not update.message.reply_to_message:
        return []

    msg = update.message.reply_to_message
    if msg.text:
        return [(msg.from_user.first_name, msg.text, datetime.now())]

    return []

async def get_recent_messages(chat_id, limit=6):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT user_name, text, time FROM messages
    WHERE chat_id = %s
    ORDER BY id DESC LIMIT %s
    """, (chat_id, limit))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    rows.reverse()
    return rows

async def get_topic_messages(user_input):
    topic = await detect_topic(user_input)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT user_name, text, time FROM messages
    WHERE topic = %s
    ORDER BY id DESC LIMIT 10
    """, (topic,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    rows.reverse()
    return rows

def format_messages(msgs):
    formatted = []
    for u, t, time in msgs:
        try:
            time_str = time.strftime("%H:%M")
        except:
            time_str = "??:??"
        formatted.append(f"[{time_str}] {u}: {t}")
    return "\n".join(formatted)

# === AI ===
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

    if is_spamming(update.message.from_user.id):
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

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

    # 🔥 MEMORY SYSTEM
    reply_msgs = await get_reply_context(update)
    recent_msgs = await get_recent_messages(update.message.chat.id)
    topic_msgs = await get_topic_messages(user_input)

    context_msgs = reply_msgs + recent_msgs + topic_msgs

    if is_why_question(user_input):
        context_msgs = filter_reason_messages(context_msgs)

    context_msgs = context_msgs[-12:]

    prompt = f"""
You are a friendly, helpful AI assistant (ChatGPT-style).

IMPORTANT:
- Detect the user's language (English or Burmese)
- Reply in the SAME language
- Be warm, natural, and slightly playful 😄
- Keep answers clear and helpful
- Use conversation context carefully

CRITICAL:
- If the user asks about time:
    → Pay attention to timestamps in the context

- If the user asks "why":
    → Find the reason from context
    → Do NOT guess
    → If no reason found, say you don’t see a clear reason

Context:
{format_messages(context_msgs)}

User:
{user_input}
"""

    try:
        response = await generate_ai(prompt)
        await update.message.reply_text(response.text)
    except errors.ClientError as e:
        if e.code == 429:
            await update.message.reply_text("⚠️ API limit reached. Try again later 🙏")
        else:
            await update.message.reply_text("⚠️ Something went wrong.")
    except:
        await update.message.reply_text("⚠️ System is busy.")

# === SUMMARY ===
async def generate_summary(days):
    conn = get_conn()
    cur = conn.cursor()

    cutoff = datetime.now() - timedelta(days=days)
    cur.execute("SELECT user_name, text FROM messages WHERE time >= %s", (cutoff,))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        return "No messages found."

    text_data = "\n".join([f"{u}: {t}" for u, t in rows])

    prompt = f"""
IMPORTANT:
- Detect language automatically

If Burmese:
Summary:
- Name: အဓိကအချက်
Decision (ရှိပါက)

If English:
Summary:
- Name: key point
Decision (if any)

Chat:
{text_data}
"""

    try:
        response = await generate_ai(prompt)
        return response.text
    except:
        return "⚠️ Could not generate summary."

# === COMMANDS ===
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

    result = "📊 Top Users:\n"
    for u, c in count.most_common(5):
        result += f"- {u}: {c}\n"

    await update.message.reply_text(result)

# === HANDLER ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_message(update)
    await chat(update, context)

# === RUN ===
if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CommandHandler("todaysummary", todaysummary))
    app.add_handler(CommandHandler("yesterdaysummary", yesterdaysummary))
    app.add_handler(CommandHandler("last3dayssummary", last3dayssummary))
    app.add_handler(CommandHandler("lastweeksummary", lastweeksummary))
    app.add_handler(CommandHandler("stats", stats))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)