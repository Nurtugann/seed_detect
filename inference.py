import os
import numpy as np
import torch
import cv2
from PIL import Image
from pycocotools import mask as mask_utils
from torch.cuda.amp import autocast
from torchvision import models, transforms
from torch import nn
from huggingface_hub import hf_hub_download

# ── Загружаем чекпоинты из локального ./model или из HF Hub ─────────────────
def get_checkpoint(filename):
    if os.path.exists(filename):
        return filename
    # иначе качаем из Hugging Face
    return hf_hub_download(
        repo_id="YOUR_USERNAME/YOUR_MODEL_REPO",
        filename=filename
    )

SAM_CHECKPOINT = get_checkpoint("sam_vit_b_01ec64.pth")
BEST_CKPT       = get_checkpoint("best_checkpoint.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Инициализируем SAM ────────────────────────────────────────────────────
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
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

# ── Инициализируем ResNet ─────────────────────────────────────────────────
resnet = models.resnet18(pretrained=True)
orig_conv = resnet.conv1
resnet.conv1 = nn.Conv2d(4, orig_conv.out_channels,
                         kernel_size=orig_conv.kernel_size,
                         stride=orig_conv.stride,
                         padding=orig_conv.padding,
                         bias=(orig_conv.bias is not None))
with torch.no_grad():
    resnet.conv1.weight[:, :3] = orig_conv.weight
    resnet.conv1.weight[:, 3:] = orig_conv.weight[:, :1]
in_f = resnet.fc.in_features
resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, 2))
resnet = resnet.to(device)
# грузим веса
ck = torch.load(BEST_CKPT, map_location=device)
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

def process_image(img_input):
    """
    img_input: либо путь к файлу, либо PIL.Image, либо numpy array (H,W,3)
    возвращает: (vis_np, seed_cnt, impurity_cnt, seed_area, impurity_area)
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        img = Image.fromarray(img_input.astype("uint8"), "RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]
    total_area = H * W

    # 1) SAM
    with torch.no_grad(), autocast():
        masks = mask_gen.generate(img_np)
    masks = sorted(masks, key=lambda m: m["area"])
    masks = [m for m in masks if m["area"] <= total_area * MAX_AREA_FRAC]

    # 2) аннотации
    anns, aid = [], 1
    for m in masks:
        rle = mask_utils.encode(
            np.asfortranarray((m["segmentation"].astype(np.uint8)*255))
        )
        rle["counts"] = rle["counts"].decode("ascii")
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
        area = m["area"]
        binm = mask_utils.decode(rle).astype(np.uint8)
        ys, xs = np.where(binm)
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)] \
               if xs.size else [0,0,0,0]
        anns.append({"id": aid, "bbox": bbox,
                     "segmentation": rle, "area": area})
        aid += 1

    # 3) фильтрация вложенных
    masks_bool = [decode_rle_bool(a["segmentation"]) for a in anns]
    drop = set()
    for i, ai in enumerate(anns):
        mi = masks_bool[i]
        x,y,w,h = ai["bbox"]
        x2,y2 = x+w-1, y+h-1
        for j, aj in enumerate(anns):
            if j==i or aj["area"]<ai["area"]: continue
            xj,yj,wj,hj = aj["bbox"]; xj2,yj2 = xj+wj-1, yj+hj-1
            if xj<=x and yj<=y and xj2>=x2 and yj2>=y2:
                if np.logical_and(mi, ~masks_bool[j]).sum()==0:
                    drop.add(ai["id"]); break
    anns = [a for a in anns if a["id"] not in drop]

    # 4) готовим батч для ResNet
    vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    crops = []
    for a in anns:
        x,y,w,h = a["bbox"]
        m_full = decode_rle_bool(a["segmentation"])[y:y+h, x:x+w]
        crop = vis[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        t_img = basic_tf(pil_img)
        np_m = (np.array(pil_img.resize((128,128), Image.NEAREST))[:,:,0]//255)
        t_m = torch.from_numpy(np_m).unsqueeze(0).float()
        crops.append(torch.cat([t_img, t_m], dim=0))
    if not crops:
        return vis, 0, 0, 0, 0

    batch = torch.stack(crops).to(device)
    resnet.eval()
    with torch.no_grad():
        preds = resnet(batch).argmax(1).cpu().numpy()

    # 5) рисуем контуры и считаем метрики
    seed_cnt = int((preds==0).sum())
    imp_cnt  = int((preds==1).sum())
    seed_area = sum(a["area"] for a,p in zip(anns,preds) if p==0)
    imp_area  = sum(a["area"] for a,p in zip(anns,preds) if p==1)
    for a,p in zip(anns,preds):
        cnts,_ = cv2.findContours(
            (decode_rle_bool(a["segmentation"])*255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = label2color[int(p)]
        cv2.drawContours(vis, cnts, -1, color, 2)

    return vis, seed_cnt, imp_cnt, seed_area, imp_area
