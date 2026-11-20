//! 生成 obb 冒烟测试的合成 DOTA 标签（真值 == 模型自身旋转框角点）。
//!
//! 跑一次 obb 模型，把每个旋转框的 4 个像素角点（`xyxyxyxy`）落成 DOTA 风格
//! `poly: [x1,y1,x2,y2,x3,y3,x4,y4]`，并写一个 COCO 同构的标签 JSON。预测 == 真值 →
//! 旋转 IoU = 1.0 → mAP = 1.0，端到端验证 obb 评估链路。
//!
//! 用法：`cargo run --example make_obb_gt`

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use serde_json::json;
use ultralytics_inference::{InferenceConfig, YOLOModel};

const MODEL: &str = "models/obb/yolo26n-obb.onnx";
const SRC: &str = "data/boats.jpg";
const OUT_IMG: &str = "data/eval/images/boats.jpg";
const OUT_JSON: &str = "data/eval/obb_labels.json";

fn main() -> Result<(), Box<dyn Error>> {
    // 与 `config/eval_obb.toml` 一致：conf=0.25 / iou=0.7 / max_det=300 / rect。
    let cfg = InferenceConfig::default()
        .with_confidence(0.25)
        .with_iou(0.7)
        .with_max_det(300)
        .with_rect(true);
    let mut model = YOLOModel::load_with_config(MODEL, cfg)?;
    let results = model.predict(Path::new(SRC))?;
    println!("任务: {} | 结果数: {}", model.task().as_str(), results.len());

    let file_name = Path::new(SRC)
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "bad source path".to_string())?
        .to_string();

    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut cat_map: BTreeMap<u64, String> = BTreeMap::new();

    for (img_id, r) in results.iter().enumerate() {
        let (h, w) = r.orig_shape;
        images.push(json!({
            "id": img_id as u64,
            "file_name": file_name,
            "width": w,
            "height": h,
        }));

        for (&id, name) in r.names.iter() {
            cat_map.entry(id as u64).or_insert_with(|| name.clone());
        }

        let Some(obb) = &r.obb else { continue };
        let corners = obb.xyxyxyxy(); // (N,4,2) 像素坐标
        let cls = obb.cls();
        for i in 0..obb.len() {
            let poly: Vec<f32> = (0..4)
                .flat_map(|j| [corners[[i, j, 0]], corners[[i, j, 1]]])
                .collect();
            annotations.push(json!({
                "image_id": img_id as u64,
                "category_id": cls[i] as u64,
                "poly": poly,
            }));
        }
    }

    let categories: Vec<_> = cat_map
        .into_iter()
        .map(|(id, name)| json!({ "id": id, "name": name }))
        .collect();

    std::fs::create_dir_all(Path::new(OUT_IMG).parent().unwrap())?;
    std::fs::copy(SRC, OUT_IMG)?;

    let dota = json!({ "images": images, "annotations": annotations, "categories": categories });
    std::fs::write(OUT_JSON, serde_json::to_string_pretty(&dota)?)?;
    println!("wrote {OUT_JSON}（{} 个 GT 旋转框）", annotations.len());
    Ok(())
}
