//! 生成 segment 冒烟测试的合成 COCO 集（真值 == 模型自身掩码）。
//!
//! 用与 `config/eval_segment.toml` 相同的推理参数跑一次 segment 模型，把每个实例的
//! 掩码（阈值 0.5 二值化）编码为**未压缩 RLE** 写进 COCO JSON。由于预测 == 真值，
//! `cargo run -- eval config/eval_segment.toml` 应得 mAP@0.5 = mAP@0.5:0.95 = 1.0，
//! 端到端验证「推理 → COCO 分割解析 → 掩码 IoU → AP」链路。
//!
//! 用法：`cargo run --example make_segment_gt`

use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use serde_json::json;
use ultralytics_inference::{InferenceConfig, YOLOModel};

const MODEL: &str = "models/seg/yolo26n-seg.onnx";
const SRC: &str = "data/bus.jpg";
const OUT_DIR: &str = "data/eval";
const OUT_JSON: &str = "data/eval/annotations_seg.json";

fn main() -> Result<(), Box<dyn Error>> {
    // 与 segment 配置一致：conf=0.25 / iou=0.7 / max_det=300 / rect。
    let cfg = InferenceConfig::default()
        .with_confidence(0.25)
        .with_iou(0.7)
        .with_max_det(300)
        .with_rect(true);

    let mut model = YOLOModel::load_with_config(MODEL, cfg)?;
    let results = model.predict(Path::new(SRC))?;
    println!("任务: {} | 结果数: {}", model.task().as_str(), results.len());

    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut cat_id: HashMap<String, i64> = HashMap::new();
    let mut categories = Vec::new();

    for (img_id, r) in results.iter().enumerate() {
        let (h, w) = r.orig_shape;
        images.push(json!({
            "id": img_id,
            "file_name": Path::new(&r.path).file_name().and_then(|s| s.to_str()),
            "width": w,
            "height": h,
        }));

        let (Some(boxes), Some(masks)) = (&r.boxes, &r.masks) else {
            continue;
        };
        let xyxy = boxes.xyxy();
        let conf = boxes.conf();
        let cls = boxes.cls();
        let n = boxes.len().min(masks.len());

        for (i, mask_view) in masks.data.outer_iter().enumerate() {
            if i >= n {
                break;
            }
            let class_id = cls[i] as usize;
            let class = r
                .names
                .get(&class_id)
                .cloned()
                .unwrap_or_else(|| class_id.to_string());

            let next_id = categories.len() as i64;
            let cid = *cat_id.entry(class.clone()).or_insert_with(|| {
                categories.push(json!({ "id": next_id, "name": class }));
                next_id
            });

            let mut bin = Vec::with_capacity(mask_view.len());
            for &v in mask_view.iter() {
                bin.push(u8::from(v > 0.5));
            }

            let x = xyxy[[i, 0]];
            let y = xyxy[[i, 1]];
            let bw = xyxy[[i, 2]] - xyxy[[i, 0]];
            let bh = xyxy[[i, 3]] - xyxy[[i, 1]];

            annotations.push(json!({
                "id": annotations.len(),
                "image_id": img_id,
                "category_id": cid,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "conf_hint": conf[i],
                "segmentation": { "size": [h, w], "counts": mask_to_rle(&bin) },
            }));
        }
    }

    let coco = json!({ "images": images, "annotations": annotations, "categories": categories });

    let out_img = PathBuf::from(OUT_DIR).join("images").join(
        Path::new(SRC)
            .file_name()
            .ok_or_else(|| "bad source path".to_string())?,
    );
    std::fs::create_dir_all(out_img.parent().unwrap())?;
    std::fs::copy(SRC, &out_img)?;
    std::fs::write(
        OUT_JSON,
        serde_json::to_string_pretty(&coco)?,
    )?;
    println!(
        "wrote {OUT_JSON} ({} images, {} GT masks)",
        images.len(),
        annotations.len()
    );
    Ok(())
}

/// 把二值掩码编码为 COCO 未压缩 RLE（`counts` 从 0 游程开始，交替 0/1）。
fn mask_to_rle(mask: &[u8]) -> Vec<u32> {
    let mut counts = Vec::new();
    let mut prev = 0u8;
    let mut run = 0u32;
    for &v in mask {
        if v == prev {
            run += 1;
        } else {
            counts.push(run);
            prev = v;
            run = 1;
        }
    }
    counts.push(run);
    counts
}
