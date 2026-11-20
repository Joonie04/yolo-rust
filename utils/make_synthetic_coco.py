"""生成用于阶段 4 冒烟测试的合成 COCO 数据集。

思路：把一次 detect 推理（conf=0.25）的 5 个框当作真值写进 COCO JSON。
由于预测 == 真值，用同一模型同一参数跑 `cargo run -- eval config/eval.toml`
应得到 mAP@0.5 ≈ 1.0，从而端到端验证「推理 → COCO 解析 → IoU 匹配 → AP」链路。

用法：python utils/make_synthetic_coco.py
"""
import json
import os
import shutil

from PIL import Image

# 来自 `cargo run -- config/detect.toml`（data/bus.jpg，conf=0.25）的归一化 xyxy。
DETECTIONS = [
    ("bus",    0.012827428057789803, 0.21351242065429688, 0.9892197847366333, 0.6922066807746887),
    ("person", 0.05817534029483795,  0.36820316314697266, 0.2934925854206085, 0.8348360061645508),
    ("person", 0.2759784162044525,   0.37682801485061646, 0.42698410153388977, 0.797484278678894),
    ("person", 0.8260728120803833,   0.35853633284568787, 1.0,                0.8149598240852356),
    ("person", 0.0,                  0.5116510391235352,  0.0738530158996582, 0.8079605102539062),
]

SRC = "data/bus.jpg"
OUT_DIR = "data/eval"
OUT_IMG = os.path.join(OUT_DIR, "images", "bus.jpg")
OUT_JSON = os.path.join(OUT_DIR, "annotations.json")

categories = [{"id": 0, "name": "person"}, {"id": 1, "name": "bus"}]
name_to_id = {c["name"]: c["id"] for c in categories}

with Image.open(SRC) as im:
    w, h = im.size

annotations = []
for i, (name, x1, y1, x2, y2) in enumerate(DETECTIONS):
    px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h
    annotations.append({
        "id": i,
        "image_id": 0,
        "category_id": name_to_id[name],
        "bbox": [px1, py1, px2 - px1, py2 - py1],
        "area": (px2 - px1) * (py2 - py1),
        "iscrowd": 0,
    })

coco = {
    "images": [{"id": 0, "file_name": "bus.jpg", "width": w, "height": h}],
    "annotations": annotations,
    "categories": categories,
}

os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
shutil.copy(SRC, OUT_IMG)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)
print(f"wrote {OUT_JSON} (image {w}x{h}, {len(annotations)} GT boxes)")
