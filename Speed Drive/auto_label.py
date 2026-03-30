import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# ======================
# 설정
# ======================
MODEL_PATH = "runs/segment/train/weights/best.pt"
IMG_DIR = "lane_dataset/images/hello"
OUT_DIR = "lane_dataset/images/auto_labeled"
CONF_THRES = 0.5

CLASS_NAMES = {
    0: "RED",
    1: "GREEN"
}

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# ======================
# 실행
# ======================
for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    results = model(img, conf=CONF_THRES, verbose=False)
    r = results[0]

    shapes = []

    if r.masks is None:
        continue

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    for i, mask in enumerate(masks):
        cls_id = classes[i]
        label = CLASS_NAMES.get(cls_id, None)
        if label is None:
            continue

        binary = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 5:
                continue

            points = cnt.squeeze().tolist()

            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageHeight": h,
        "imageWidth": w,
        "imageData": None
    }

    with open(os.path.join(OUT_DIR, img_name.replace(".jpg", ".json")), "w") as f:
        json.dump(labelme_json, f, indent=2)

    print(f"[AUTO] {img_name}")
