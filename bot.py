import os
import logging
import sqlite3
import asyncio
from datetime import datetime, timedelta
from collections import Counter

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing BOT_TOKEN or GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# === DATABASE ===
conn = sqlite3.connect("chat.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    text TEXT,
    time TEXT
)
""")
conn.commit()

# === CACHE ===
BOT_USERNAME = None
user_last_message = {}

async def get_bot_username(context):
    global BOT_USERNAME
    if not BOT_USERNAME:
        bot = await context.bot.get_me()
        BOT_USERNAME = bot.username.lower()
    return BOT_USERNAME

# === SAVE MESSAGE ===
async def save_message(update: Update):
    if update.message and update.message.text:
        cursor.execute(
            "INSERT INTO messages (user, text, time) VALUES (?, ?, ?)",
            (
                update.message.from_user.first_name,
                update.message.text,
                datetime.now().isoformat()
            )
        )
        conn.commit()

# === DATA FUNCTIONS ===
def get_recent(limit=50):
    cursor.execute("SELECT user, text FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    rows.reverse()
    return rows

def filter_important(messages):
    keywords = ["important", "decide", "plan", "meeting", "deadline", "?"]
    return [m for m in messages if any(k in m[1].lower() for k in keywords)]

def format_messages(msgs):
    return "\n".join([f"{u}: {t}" for u, t in msgs])

# === SPAM PROTECTION ===
def is_spamming(user_id):
    now = datetime.now().timestamp()
    if user_id in user_last_message and now - user_last_message[user_id] < 2:
        return True
    user_last_message[user_id] = now
    return False

# === AI CALL (ASYNC FIX) ===
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

    # 🚫 Anti-spam
    if is_spamming(user_id):
        return

    user_input = update.message.text
    bot_username = await get_bot_username(context)

    # 💬 Reply OR mention logic
    is_reply = (
        update.message.reply_to_message and
        update.message.reply_to_message.from_user.id == context.bot.id
    )

    if update.message.chat.type != "private":
        if not is_reply and f"@{bot_username}" not in user_input.lower():
            return

    logger.info(f"Incoming: {user_input}")

    # ⌨️ Typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    recent = get_recent(50)
    important = filter_important(recent)

    prompt = f"""
You are a helpful AI assistant in a Telegram group.

Rules:
- Always respond in Burmese
- Be concise and friendly

Important context:
{format_messages(important)}

User:
{user_input}
"""

    try:
        response = await generate_ai(prompt)
        await update.message.reply_text(response.text)

    except Exception as e:
        logger.error(e)
        await update.message.reply_text("⚠️ AI မရရှိနိုင်ပါ။")

# === SUMMARY ===
async def summarize(update, msgs):
    if not msgs:
        await update.message.reply_text("စာမတွေ့ပါ။")
        return

    prompt = f"""
အောက်ပါ chat ကို အကျဉ်းချုပ်ပေးပါ။

Chat:
{format_messages(msgs)}
"""

    try:
        response = await generate_ai(prompt)
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("⚠️ AI မရရှိနိုင်ပါ။")

def get_messages_by_days(days):
    cutoff = datetime.now() - timedelta(days=days)
    cursor.execute("SELECT user, text, time FROM messages")
    rows = cursor.fetchall()
    return [(u, t) for u, t, time in rows if datetime.fromisoformat(time) >= cutoff]

# === COMMANDS ===
async def todaysummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await summarize(update, get_messages_by_days(1))

async def lastweeksummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await summarize(update, get_messages_by_days(7))

# === STATS ===
def get_top_users(limit=100):
    cursor.execute("SELECT user FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    users = [row[0] for row in cursor.fetchall()]
    count = Counter(users)

    result = "📊 Top Users:\n"
    for user, c in count.most_common(5):
        result += f"- {user}: {c} messages\n"
    return result

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_top_users())

# === HANDLER ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_message(update)
    await chat(update, context)

# === BUILD APP ===
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
app.add_handler(CommandHandler("stats", stats))
app.add_handler(CommandHandler("todaysummary", todaysummary))
app.add_handler(CommandHandler("lastweeksummary", lastweeksummary))

logger.info("Bot is running...")

# === RUN ===
app.run_polling(drop_pending_updates=True)