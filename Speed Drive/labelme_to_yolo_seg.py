import os
import json
import cv2

IMG_DIR = "lane_dataset/images/mission3"
LABEL_OUT = "lane_dataset/labels/mission3"

os.makedirs(LABEL_OUT, exist_ok=True)

LABEL_MAP = {
    "RED": 0,
    "GREEN": 1
}

for file in os.listdir(IMG_DIR):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(IMG_DIR, file)
    with open(json_path, "r") as f:
        data = json.load(f)

    img_path = os.path.join(IMG_DIR, data["imagePath"])
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    txt_name = file.replace(".json", ".txt")
    with open(os.path.join(LABEL_OUT, txt_name), "w") as out_txt:

        for shape in data["shapes"]:
            label = shape["label"]

            if label not in LABEL_MAP:
                continue

            class_id = LABEL_MAP[label]
            line = str(class_id)

            for x, y in shape["points"]:
                line += f" {x/w:.6f} {y/h:.6f}"

            out_txt.write(line + "\n")
