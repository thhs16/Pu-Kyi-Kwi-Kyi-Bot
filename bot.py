from datetime import datetime, timedelta
import sqlite3
from collections import Counter
import os
import logging

from telegram import Update
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

# === SAVE MESSAGE ===
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

# === GET RECENT ===
def get_recent(limit=50):
    cursor.execute(
        "SELECT user, text FROM messages ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    rows.reverse()
    return rows

# === FORMAT ===
def format_messages(msgs):
    return "\n".join([f"{u}: {t}" for u, t in msgs])

# === CHAT FUNCTION ===
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_input = update.message.text

    # 🔥 FIX: safely get bot username
    bot = await context.bot.get_me()
    bot_username = bot.username.lower()

    logger.info(f"Incoming message: {user_input}")
    logger.info(f"Bot username: @{bot_username}")

    # Only respond if mentioned in group
    if update.message.chat.type != "private":
        if f"@{bot_username}" not in user_input.lower():
            logger.info("Bot not mentioned, ignoring...")
            return

    recent = get_recent(50)

    prompt = f"""
You are a smart AI assistant.

IMPORTANT:
- Always respond in Burmese
- Friendly tone

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
        logger.error(f"AI ERROR: {e}")
        await update.message.reply_text("⚠️ AI မရရှိနိုင်ပါ။")

# === ANALYTICS ===
def get_top_users(limit=100):
    cursor.execute(
        "SELECT user FROM messages ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    users = [row[0] for row in cursor.fetchall()]
    count = Counter(users)

    result = "📊 အများဆုံး ပြောသူများ:\n"
    for user, c in count.most_common(5):
        result += f"- {user}: {c} စာ\n"

    return result

# === COMMANDS ===
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_top_users())

# === COMBINED HANDLER ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_message(update, context)
    await chat(update, context)

# === BUILD APP ===
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
app.add_handler(CommandHandler("stats", stats))

logger.info("Bot is starting...",)

# === RUN ===
app.run_polling(drop_pending_updates=True)