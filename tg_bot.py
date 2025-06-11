import os
import io
import base64
import httpx
from PIL import Image
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# 📌 Не забудь в окружении задать:
#   BOT_TOKEN – токен от BotFather
#   ML_SERVICE_URL – https://huggingface.co/spaces/USERNAME/REPO_NAME/run/predict
# (если Space приватный, ещё HF_TOKEN)

BOT_TOKEN      = os.environ["BOT_TOKEN"]
ML_SERVICE_URL = os.environ["ML_SERVICE_URL"]
HF_TOKEN       = os.environ.get("HF_TOKEN")  # опционально

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Пришли фото — отмечу семена и верну результат."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    f     = await photo.get_file()
    in_p  = f"{photo.file_id}.jpg"
    await f.download_to_drive(in_p)

    # 1) Загружаем картинку и кодируем её в base64
    img = Image.open(in_p)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {"data": [b64]}
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # 2) Шлём на HuggingFace Space
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            ML_SERVICE_URL,
            json=payload,
            headers=headers,
            timeout=120.0
        )
    resp.raise_for_status()
    out_b64, stats = resp.json()["data"]

    # 3) Декодируем ответ и сохраняем
    out_bytes = base64.b64decode(out_b64)
    out_p = f"out_{photo.file_id}.png"
    with open(out_p, "wb") as fo:
        fo.write(out_bytes)

    # 4) Отправляем пользователю
    await update.message.reply_photo(photo=open(out_p, "rb"))
    await update.message.reply_text(stats)

    # 5) Чистим временные файлы
    for p in (in_p, out_p):
        try:
            os.remove(p)
        except OSError:
            pass

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("🚀 Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
