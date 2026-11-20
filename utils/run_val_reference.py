#!/usr/bin/env python
"""跑 Ultralytics model.val() 作为等价性参考，落盘汇总 + 每类 AP。

用法:
    python utils/run_val_reference.py <model.pt> <data.yaml> <out.json>

对齐参数与 Rust eval 一致：conf=0.001, iou=0.7, imgsz=640, batch=1, device=cpu。
输出 JSON 含 mAP@0.5 / mAP@0.5:0.95 / precision / recall / per_class_ap50 / per_class_ap。
"""
import json
import os
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1

    model_pt, data_yaml, out_json = sys.argv[1:4]
    device = os.environ.get("EQUIV_DEVICE", "cpu")
    batch = int(os.environ.get("EQUIV_BATCH", "1"))
    m = YOLO(model_pt)
    res = m.val(
        data=data_yaml,
        conf=0.001,
        iou=0.7,
        imgsz=640,
        batch=batch,
        device=device,
        verbose=True,
        plots=False,
    )

    rd = res.results_dict
    box = res.box
    out = {
        "ultralytics_version": ultralytics.__version__,
        "model": Path(model_pt).name,
        "data": data_yaml,
        "params": {"conf": 0.001, "iou": 0.7, "imgsz": 640, "batch": batch, "device": device},
        "map50": float(rd.get("metrics/mAP50(B)", float("nan"))),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", float("nan"))),
        "precision": float(rd.get("metrics/precision(B)", float("nan"))),
        "recall": float(rd.get("metrics/recall(B)", float("nan"))),
        "per_class_ap50": [float(x) for x in getattr(box, "ap50", [])],
        "per_class_ap": [float(x) for x in getattr(box, "ap", [])],
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[dump] {out_json}")
    print(f"[dump] mAP@0.5={out['map50']:.5f}  mAP@0.5:0.95={out['map50_95']:.5f}  "
          f"P={out['precision']:.5f}  R={out['recall']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
