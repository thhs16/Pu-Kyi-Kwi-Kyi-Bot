from datetime import datetime, timedelta
import sqlite3
from collections import Counter
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai

# === CONFIG ===
import os

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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

# === GET BY DAYS ===
def get_messages_by_days(days=1):
    cutoff = datetime.now() - timedelta(days=days)

    cursor.execute("SELECT user, text, time FROM messages")
    rows = cursor.fetchall()

    return [
        (u, t)
        for u, t, time in rows
        if datetime.fromisoformat(time) >= cutoff
    ]

# === FIXED YESTERDAY ===
def get_yesterday_messages():
    today = datetime.now().date()
    start = datetime.combine(today - timedelta(days=1), datetime.min.time())
    end = datetime.combine(today, datetime.min.time())

    cursor.execute("SELECT user, text, time FROM messages")
    rows = cursor.fetchall()

    return [
        (u, t)
        for u, t, time in rows
        if start <= datetime.fromisoformat(time) < end
    ]

# === SMART SEARCH ===
def search_messages(query, limit=30):
    words = query.lower().split()
    results = []

    for word in words:
        cursor.execute(
            "SELECT user, text FROM messages WHERE LOWER(text) LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{word}%", limit)
        )
        results.extend(cursor.fetchall())

    return list(set(results))[:limit]

# === IMPORTANT FILTER ===
def filter_important(messages):
    keywords = ["decide", "important", "agree", "plan", "meeting", "deadline", "?"]
    return [
        (u, t)
        for u, t in messages
        if any(k in t.lower() for k in keywords)
    ][:50]

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

# === FORMAT ===
def format_messages(msgs):
    return "\n".join([f"{u}: {t}" for u, t in msgs])

# === SUMMARIZE (BURMESE) ===
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
        print("ERROR:", e)
        return "⚠️ AI မရရှိနိုင်ပါ။"

# === SUMMARY COMMANDS ===
async def todaysummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msgs = get_messages_by_days(1)
    await update.message.reply_text(summarize(format_messages(msgs)))

async def yesterdaysummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msgs = get_yesterday_messages()
    await update.message.reply_text(summarize(format_messages(msgs)))

async def last3dayssummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msgs = get_messages_by_days(3)
    await update.message.reply_text(summarize(format_messages(msgs)))

async def lastweeksummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msgs = get_messages_by_days(7)
    await update.message.reply_text(summarize(format_messages(msgs)))

# === AI CHAT (GROUP ONLY) ===
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_input = update.message.text
    bot_username = context.bot.username.lower()

    # Only respond if mentioned
    if f"@{bot_username}" not in user_input.lower():
        return

    recent = get_recent(50)
    relevant = search_messages(user_input, 30)
    important = filter_important(recent)

    prompt = f"""
You are a smart AI assistant.

IMPORTANT:
- Always respond in Burmese
- Friendly tone

If question is about chat → use history
If general → answer normally

Recent:
{format_messages(recent)}

Relevant:
{format_messages(relevant)}

Important:
{format_messages(important)}

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
        print("ERROR:", e)
        await update.message.reply_text("⚠️ AI မရရှိနိုင်ပါ။")

# === ANALYTICS ===
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_top_users())

# === COMBINED HANDLER ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await save_message(update, context)
    await chat(update, context)

# === RUN ===
app = ApplicationBuilder().token(BOT_TOKEN).build()

# ONE handler (important fix)
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

# Commands
app.add_handler(CommandHandler("todaysummary", todaysummary))
app.add_handler(CommandHandler("yesterdaysummary", yesterdaysummary))
app.add_handler(CommandHandler("last3dayssummary", last3dayssummary))
app.add_handler(CommandHandler("lastweeksummary", lastweeksummary))
app.add_handler(CommandHandler("stats", stats))

print("Bot is running...")
app.run_polling()