import nest_asyncio
nest_asyncio.apply()

import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
from pycocotools import mask as mask_utils
from torch.cuda.amp import autocast
from torchvision import models, transforms
from torch import nn
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
import torchvision.ops.boxes as box_ops
import traceback

# ── ПАТЧ ДЛЯ batched_nms ─────────────────────────────────────────────────────────
if not hasattr(box_ops, "_orig_batched_nms"):
    box_ops._orig_batched_nms = box_ops.batched_nms

def _patched_batched_nms(boxes, scores, idxs, iou_threshold):
    orig_fn = box_ops._orig_batched_nms
    if idxs.device != boxes.device:
        idxs = idxs.to(boxes.device)
    return orig_fn(boxes, scores, idxs, iou_threshold)

box_ops.batched_nms = _patched_batched_nms

# ── ФИКС СИДОВ ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── ИНИЦИАЛИЗАЦИЯ SAM ───────────────────────────────────────────────────
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
for name, p in sam.named_parameters():
    if "mask_decoder" not in name:
        p.requires_grad = False


mask_gen = SamAutomaticMaskGenerator(
    sam,
    points_per_side=64,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.85,
    box_nms_thresh=0.3,
    min_mask_region_area=0,
    crop_n_layers=1,
    crop_nms_thresh=0.3,
    crop_overlap_ratio=0.5,
    crop_n_points_downscale_factor=1,
    output_mode="binary_mask"
)
MAX_AREA_FRAC = 0.005

# ── ИНИЦИАЛИЗАЦИЯ ResNet ───────────────────────────────────────────────
RES_CKPT_DIR = "res"
resnet = models.resnet18(pretrained=True)
orig_conv = resnet.conv1
resnet.conv1 = nn.Conv2d(
    4, orig_conv.out_channels,
    kernel_size=orig_conv.kernel_size,
    stride=orig_conv.stride,
    padding=orig_conv.padding,
    bias=(orig_conv.bias is not None)
)
with torch.no_grad():
    resnet.conv1.weight[:, :3] = orig_conv.weight
    resnet.conv1.weight[:, 3:] = orig_conv.weight[:, :1]

in_f = resnet.fc.in_features
resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, 2))
resnet = resnet.to(device)

CHK = os.path.join(RES_CKPT_DIR, "best_checkpoint.pth")
if os.path.exists(CHK):
    ck = torch.load(CHK, map_location=device)
    try:
        resnet.load_state_dict(ck["model_state_dict"])
    except RuntimeError:
        resnet.load_state_dict(ck["model_state_dict"], strict=False)
        sd = ck["model_state_dict"]
        resnet.fc[1].weight.data.copy_(sd["fc.weight"])
        resnet.fc[1].bias.data.copy_(sd["fc.bias"])

basic_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
label2color = {0: (0, 255, 0), 1: (128, 0, 128)}

def decode_rle_bool(rle):
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
    return mask_utils.decode(rle).astype(bool)

# ── ПАЙПЛАЙН ДЛЯ ОДНОГО ИЗОБРАЖЕНИЯ ─────────────────────────────────
def process_image(input_path: str):
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]
    total_area = H * W

    # 1) Генерация масок SAM
    with torch.no_grad(), autocast():
        masks = mask_gen.generate(img_np)
    masks = sorted(masks, key=lambda m: m["area"])
    masks = [m for m in masks if m["area"] <= total_area * MAX_AREA_FRAC]

    # 2) Сбор аннотаций
    anns, aid = [], 1
    for m in masks:
        rle = mask_utils.encode(np.asfortranarray((m["segmentation"].astype(np.uint8)*255)))
        rle["counts"] = rle["counts"].decode("ascii")
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
        area = m["area"]
        binm = mask_utils.decode(rle).astype(np.uint8)
        ys, xs = np.where(binm)
        if xs.size and ys.size:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            bbox = [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)]
        else:
            bbox = [0, 0, 0, 0]
        anns.append({"id": aid, "bbox": bbox, "segmentation": rle, "area": area})
        aid += 1

    # 3) Удаление вложенных
    areas = [a["area"] for a in anns]
    masks_bool = [decode_rle_bool(a["segmentation"]) for a in anns]
    drop = set()
    for i, ai in enumerate(anns):
        mi = masks_bool[i]
        x1, y1, w_i, h_i = ai["bbox"]
        x2, y2 = x1+w_i-1, y1+h_i-1
        for j, aj in enumerate(anns):
            if j==i or areas[j]<areas[i]: continue
            xj1, yj1, w_j, h_j = aj["bbox"]
            xj2, yj2 = xj1+w_j-1, yj1+h_j-1
            if not (xj1<=x1 and yj1<=y1 and xj2>=x2 and yj2>=y2): continue
            if np.logical_and(mi, ~masks_bool[j]).sum()==0:
                drop.add(ai["id"])
                break
    anns = [a for a in anns if a["id"] not in drop]

    # 4) Подготовка виз
    vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    crops = []
    for a in anns:
        x, y, w, h = a["bbox"]
        m_full = decode_rle_bool(a["segmentation"])[y:y+h, x:x+w]
        crop = vis[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pil_m   = Image.fromarray((m_full*255).astype(np.uint8))
        t_img   = basic_tf(pil_img)
        np_m    = np.array(pil_m.resize((128,128), Image.NEAREST))//255
        t_m     = torch.from_numpy(np_m).unsqueeze(0).float()
        crops.append(torch.cat([t_img, t_m], dim=0))
    if not crops:
        return vis, 0, 0

    # 5) Классификация
    batch = torch.stack(crops).to(device)
    resnet.eval()
    with torch.no_grad():
        preds = resnet(batch).argmax(dim=1).cpu().numpy()

    # 6) Рисуем контуры
    for a, p in zip(anns, preds):
        m_full = decode_rle_bool(a["segmentation"])
        cnts,_ = cv2.findContours((m_full*255).astype(np.uint8),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, label2color[p], 2)

    seed_cnt     = int((preds == 0).sum())
    impurity_cnt = int((preds == 1).sum())

    # Подсчёт площади
    seed_area     = sum(a["area"] for a, p in zip(anns, preds) if p == 0)
    impurity_area = sum(a["area"] for a, p in zip(anns, preds) if p == 1)

    return vis, seed_cnt, impurity_cnt, seed_area, impurity_area

# ── TELEGRAM-ОБЁРТКА ──────────────────────────────────────────────────────────
BOT_TOKEN = "7951372047:AAFmK6M5vb2-rPi2dfCgA60fMV-IJzKJx6Q"
TMP_DIR   = "tmp_telegram"
os.makedirs(TMP_DIR, exist_ok=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Привет! Пришли фото — отмечу семена и верну результат.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    f     = await photo.get_file()
    in_p  = os.path.join(TMP_DIR, f"{photo.file_id}.jpg")
    out_p = os.path.join(TMP_DIR, f"out_{photo.file_id}.jpg")
    await f.download_to_drive(in_p)

    try:
        # Шаг 1: генерация масок
        await update.message.reply_text("🔍 Генерирую маски с помощью SAM...")
        await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

        # Шаг 2: фильтрация и подготовка
        await update.message.reply_text("⚙️ Фильтрую и подготавливаю данные...")
        await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

        # Шаг 3: классификация
        await update.message.reply_text("🤖 Классифицирую семена и примеси...")
        await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)

        # Выполняем полный пайплайн, теперь process_image отдаёт
        #  (vis, seed_cnt, impurity_cnt, seed_area, impurity_area)
        vis, seed_cnt, impurity_cnt, seed_area, impurity_area = process_image(in_p)

        # Подсчитываем итоги
        total_cnt  = seed_cnt + impurity_cnt
        total_area = seed_area + impurity_area

        pct_cnt_seed     = seed_cnt     / total_cnt  * 100 if total_cnt>0  else 0.0
        pct_cnt_impurity = impurity_cnt / total_cnt  * 100 if total_cnt>0  else 0.0
        pct_area_seed     = seed_area     / total_area * 100 if total_area>0 else 0.0
        pct_area_impurity = impurity_area / total_area * 100 if total_area>0 else 0.0

        # Сохраняем и отправляем результат
        cv2.imwrite(out_p, vis)
        await update.message.reply_photo(photo=open(out_p, "rb"))

        # Финальная статистика
        await update.message.reply_text(
            "✅ Готово!\n\n"
            "📊 Результаты по количеству объектов:\n"
            f"  • Всего: {total_cnt}\n"
            f"  • Семян: {seed_cnt} ({pct_cnt_seed:.1f}%)\n"
            f"  • Примесей: {impurity_cnt} ({pct_cnt_impurity:.1f}%)\n\n"
            "📐 Результаты по суммарной площади (пиксели):\n"
            f"  • Всего: {total_area}\n"
            f"  • Семян: {seed_area} ({pct_area_seed:.1f}%)\n"
            f"  • Примесей: {impurity_area} ({pct_area_impurity:.1f}%)"
        )

    except RecursionError:
        tb = traceback.format_exc()
        print(f"[ERROR] RecursionError:\n{tb}")
        await update.message.reply_text("❌ Ошибка: переполнение глубины рекурсии.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Exception:\n{tb}")
        await update.message.reply_text(f"❌ Ошибка при обработке:\n{e}")
    finally:
        for p in (in_p, out_p):
            if os.path.exists(p):
                os.remove(p)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🚀 Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()