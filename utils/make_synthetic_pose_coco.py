"""生成 pose 冒烟测试的合成 COCO 集（真值 == 模型自身关键点）。

读取 `cargo run -- config/pose.toml` 产出的 results.json，把模型的 boxes/keypoints
当作真值写进 COCO JSON（关键点转像素、可见性 v=2）。预测 == 真值 → OKS=1.0 → mAP=1.0，
端到端验证「推理 → COCO 关键点解析 → OKS 匹配 → AP」链路。

用法：python utils/make_synthetic_pose_coco.py results/pose/<时间戳>/results.json
"""
import json
import os
import shutil
import sys

OUT_DIR = "data/eval"
OUT_JSON = os.path.join(OUT_DIR, "annotations_pose.json")


def main(results_path: str) -> None:
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    result = data["results"][0]
    orig_h, orig_w = result["orig_shape"]  # [height, width]
    src = result["path"]
    out_img = os.path.join(OUT_DIR, "images", os.path.basename(src))

    annotations = []
    for ann_id, (det, person) in enumerate(
        zip(result["detections"], result["keypoints"]["keypoints"])
    ):
        b = det["box"]
        x = b["x1"] * orig_w
        y = b["y1"] * orig_h
        w = (b["x2"] - b["x1"]) * orig_w
        h = (b["y2"] - b["y1"]) * orig_h
        keypoints = []
        for kx, ky, _kc in person:
            keypoints += [kx * orig_w, ky * orig_h, 2]  # v=2（可见）

        annotations.append({
            "id": ann_id,
            "image_id": 0,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "keypoints": keypoints,
            "num_keypoints": 17,
        })

    coco = {
        "images": [{
            "id": 0,
            "file_name": os.path.basename(src),
            "width": orig_w,
            "height": orig_h,
        }],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    shutil.copy(src, out_img)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_JSON} ({len(annotations)} GT persons, {orig_w}x{orig_h})")


if __name__ == "__main__":
    main(sys.argv[1])
