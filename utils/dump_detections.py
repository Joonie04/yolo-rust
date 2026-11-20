#!/usr/bin/env python
"""把 Ultralytics .pt 的 detect 预测落盘为归一化检测（等价性验证的 Python 参考）。

用法:
    python utils/dump_detections.py <model.pt> <out.json> <img1> [img2 ...]

对齐参数与 Rust 侧一致：conf=0.001, iou=0.7, imgsz=640, max_det=300。
检测框用归一化 xyxy（xyxyn），与 Rust 结果 JSON 的 `box` 字段同口径。
"""
import json
import os
import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1

    model_pt, out_json, *imgs = sys.argv[1:]
    conf = float(os.environ.get("EQUIV_CONF", "0.001"))
    m = YOLO(model_pt)
    out = {
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "model": Path(model_pt).name,
        "params": {"conf": conf, "iou": 0.7, "imgsz": 640, "max_det": 300, "device": "cpu"},
        "images": {},
    }

    for img in imgs:
        r = m.predict(
            img, conf=conf, iou=0.7, imgsz=640, max_det=300, device="cpu", verbose=False
        )[0]
        boxes = r.boxes
        name = Path(img).name
        if boxes is None or len(boxes) == 0:
            out["images"][name] = []
            continue
        xyxyn = boxes.xyxyn.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        dets = []
        for (x1, y1, x2, y2), c, k in zip(xyxyn, scores, cls):
            k = int(k)
            dets.append(
                {
                    "name": m.names[k],
                    "class": k,
                    "confidence": float(c),
                    "box": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                }
            )
        out["images"][name] = dets

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n = sum(len(v) for v in out["images"].values())
    print(f"dumped {len(out['images'])} images, {n} detections -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
