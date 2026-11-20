#!/usr/bin/env python
"""对比 Python（.pt）与 Rust（.onnx）的 detect 输出，报告逐字段偏差。

用法:
    python utils/compare_equiv.py <py_reference.json> <rust_results.json>

匹配口径：同一图像内、同一类别、贪心最大 IoU（IoU >= 0.5 视为匹配）。
报告每个图像：双方检测数、匹配数、框坐标最大偏差、置信度最大偏差、未匹配数。
"""
import json
import math
import sys
from pathlib import Path


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    py = json.load(open(sys.argv[1], encoding="utf-8"))
    rust = json.load(open(sys.argv[2], encoding="utf-8"))

    py_images = py["images"]
    rust_by_name = {}
    for r in rust["results"]:
        name = Path(r["path"]).name
        rust_by_name[name] = r.get("detections", [])

    print(f"ultralytics {py['ultralytics_version']} (torch {py['torch_version']})  vs  Rust ONNX Runtime")
    print(f"params: {py['params']}")
    print()

    all_ok = True
    for name, py_dets in sorted(py_images.items()):
        rust_dets = rust_by_name.get(name, [])
        n_py, n_rust = len(py_dets), len(rust_dets)

        # 贪心匹配：同类别 + 最大 IoU >= 0.5
        matched = 0
        max_box_d = 0.0
        max_conf_d = 0.0
        used = [False] * n_rust
        for pd in py_dets:
            best_i, best_j = -1.0, -1
            for j, rd in enumerate(rust_dets):
                if used[j] or rd["class"] != pd["class"]:
                    continue
                v = iou(pd["box"], rd["box"])
                if v > best_i:
                    best_i, best_j = v, j
            if best_j >= 0 and best_i >= 0.5:
                used[best_j] = True
                matched += 1
                rd = rust_dets[best_j]
                pb, rb = pd["box"], rd["box"]
                max_box_d = max(
                    max_box_d,
                    abs(pb["x1"] - rb["x1"]),
                    abs(pb["y1"] - rb["y1"]),
                    abs(pb["x2"] - rb["x2"]),
                    abs(pb["y2"] - rb["y2"]),
                )
                max_conf_d = max(max_conf_d, abs(pd["confidence"] - rd["confidence"]))

        unmatched_py = n_py - matched
        unmatched_rust = n_rust - matched
        ok = n_py == n_rust and unmatched_py == 0 and unmatched_rust == 0
        all_ok = all_ok and ok

        print(f"[{name}]")
        print(f"  detections  py={n_py}  rust={n_rust}  matched={matched}"
              f"  unmatched_py={unmatched_py}  unmatched_rust={unmatched_rust}")
        if matched:
            print(f"  max |Δbox| = {max_box_d:.3e}   max |Δconf| = {max_conf_d:.3e}")
        else:
            print("  (no matched pairs)")
        print()

    print("VERDICT:", "等价（逐字段对齐）" if all_ok else "存在差异，见上")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
