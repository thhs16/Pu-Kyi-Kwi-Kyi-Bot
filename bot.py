import os
import logging
import sqlite3
from datetime import datetime, timedelta
from collections import Counter
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Config / Secrets ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN or not GEMINI_API_KEY:
    logger.error("Missing BOT_TOKEN or GEMINI_API_KEY")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

# === SQLite persistent on Fly volume ===
DB_PATH = "/data/chat.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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

# === Save Message ===
async def save_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        user = update.message.from_user.first_name
        text = update.message.text
        time = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO messages (user, text, time) VALUES (?, ?, ?)",
            (user, text, time)
        )
        conn.commit()

# === Utilities ===
def get_recent(limit=50):
    cursor.execute("SELECT user, text FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    rows.reverse()
    return rows

def format_messages(msgs):
    return "\n".join([f"{u}: {t}" for u, t in msgs])

# === Summarize / AI Functions ===
def summarize(messages):
    if not messages:
        return "စာမတွေ့ပါ။"
    prompt = f"""
အောက်ပါ chat ကို WHO SAID WHAT အလိုက် အကျဉ်းချုပ်ပေးပါ။

IMPORTANT:
- မြန်မာဘာသာဖြင့် ပြန်ပါ
- နားလည်လွယ်အောင် ရေးပါ

Format:
Summary:
- Name: အဓိကအချက်

ပြီးလျှင်:
Decision (ရှိပါက)

Chat:
{messages}
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error("AI ERROR: %s", e)
        return "⚠️ AI မရရှိနိုင်ပါ။"

# === Handlers ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_message(update, context)

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_input = update.message.text
    bot_username = context.bot.username.lower()
    if f"@{bot_username}" not in user_input.lower():
        return

    recent = get_recent(50)
    prompt = f"""
You are a smart AI assistant.
Recent:
{format_messages(recent)}
User question:
{user_input}
Answer in Burmese:
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error("AI ERROR: %s", e)
        await update.message.reply_text("⚠️ AI မရရှိနိုင်ပါ။")

# === Summary Commands ===
async def todaysummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msgs = get_recent(50)
    await update.message.reply_text(summarize(format_messages(msgs)))

# === Build Application ===
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
app.add_handler(CommandHandler("todaysummary", todaysummary))

# === Startup Logging ===
logger.info("Bot is starting...", flush=True)

# === Run Bot ===
app.run_polling(drop_pending_updates=True)