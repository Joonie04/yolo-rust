//! 生成 semantic 冒烟测试的合成标签集（真值 == 模型自身语义类别图）。
//!
//! 跑一次 semantic 模型，把整图语义类别图（`Array2<u16>`，类别 id < 256）落成 8 位
//! 灰度 PNG，并写一个 `{"images":[{"file","label"}]}` 标签 JSON。预测 == 真值 →
//! 逐类 IoU = 1.0 → mIoU = 1.0，端到端验证 semantic 评估链路。
//!
//! 用法：`cargo run --example make_semantic_gt`

use std::error::Error;
use std::path::{Path, PathBuf};

use serde_json::json;
use ultralytics_inference::{InferenceConfig, YOLOModel};

const MODEL: &str = "models/sem/yolo26n-sem.onnx";
const SRC: &str = "data/bus.jpg";
const OUT_DIR: &str = "data/eval";
const OUT_JSON: &str = "data/eval/semantic_labels.json";

fn main() -> Result<(), Box<dyn Error>> {
    let mut model = YOLOModel::load_with_config(MODEL, InferenceConfig::default())?;
    let results = model.predict(Path::new(SRC))?;
    println!("任务: {} | 结果数: {}", model.task().as_str(), results.len());

    let file_name = Path::new(SRC)
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "bad source path".to_string())?
        .to_string();
    let label_name = format!(
        "{}_sem.png",
        Path::new(&file_name).file_stem().and_then(|s| s.to_str()).unwrap_or("label")
    );

    let out_img = PathBuf::from(OUT_DIR).join("images").join(&file_name);
    let out_label = PathBuf::from(OUT_DIR).join("images").join(&label_name);

    let mut labeled = 0usize;
    for r in &results {
        let Some(sem) = &r.semantic_mask else { continue };
        let (h, w) = (sem.data.nrows(), sem.data.ncols());
        // 类别 id < 256（COCO-stuff 类目），u16 → u8 无损。
        let data: Vec<u8> = sem.data.iter().map(|&v| v as u8).collect();
        let gray = image::GrayImage::from_raw(w as u32, h as u32, data)
            .ok_or_else(|| "semantic mask 尺寸与数据长度不一致".to_string())?;
        std::fs::create_dir_all(out_label.parent().unwrap())?;
        gray.save(&out_label)?;
        labeled += 1;
    }

    std::fs::copy(SRC, &out_img)?;
    let labels = json!({ "images": [{ "file": file_name, "label": label_name }] });
    std::fs::write(OUT_JSON, serde_json::to_string_pretty(&labels)?)?;
    println!("wrote {OUT_JSON} + {label_name} ({labeled} 张语义标签图)");
    Ok(())
}
