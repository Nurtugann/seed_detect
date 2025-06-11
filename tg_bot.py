import os, httpx, base64, io
from PIL import Image

ML_URL = os.environ["ML_SERVICE_URL"]  # "https://huggingface.co/spaces/Nurtugan/seed_detect/run/predict"

async def handle_photo(update, context):
    photo = update.message.photo[-1]
    f = await photo.get_file()
    in_p = f"{photo.file_id}.jpg"
    await f.download_to_drive(in_p)

    # кодируем в base64
    img = Image.open(in_p)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {"data": [b64]}
    resp = httpx.post(ML_URL, json=payload, timeout=120.0)
    data = resp.json()["data"]  # это список из двух элементов: [out_image_b64, stats_text]

    out_b64, stats = data
    out_bytes = base64.b64decode(out_b64)
    out_p = f"out_{photo.file_id}.png"
    with open(out_p, "wb") as fo:
        fo.write(out_bytes)

    # шлём результат в чат
    await update.message.reply_photo(open(out_p, "rb"))
    await update.message.reply_text(stats)
