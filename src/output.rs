//! 结果输出：控制台摘要、标注图、JSON。
//!
//! JSON 采用「公共信封 + 任务载荷」的结构：`path` / `task` / `orig_shape` /
//! `inference_shape` / `speed_ms` 每个结果都有；`detections` / `masks` /
//! `keypoints` / `obb` / `probs` / `semantic` / `depth` 按任务按需出现。
//! 坐标一律归一化到 `[0, 1]`（掩码面积、像素占比等天然尺度无关）。

use std::collections::HashMap;
use std::path::PathBuf;

use image::DynamicImage;
use serde_json::{json, Value};
use ultralytics_inference::{
    Boxes, DepthMap, Keypoints, Masks, Obb, Probs, Results, SemanticMask, SourceMeta,
};
#[cfg(feature = "annotate")]
use ultralytics_inference::annotate::annotate_image;
use ultralytics_inference::io::SaveResults;

use crate::config::OutputConfig;
use crate::error::Result;

/// 输出器：把每个 `Results` 落地为控制台摘要 / 标注图 / JSON。
pub struct Output {
    /// 输出目录（已含 runs/<task>/<时间戳> 子目录）。
    dir: PathBuf,
    /// 是否保存标注图。
    save_annotated: bool,
    /// 是否保存 JSON。
    save_json: bool,
    /// 是否打印每张图摘要。
    verbose: bool,
    /// 当前任务名（detect/segment/semantic/depth/classify/pose/obb）。
    task: String,
    /// 输入源是否为视频/流（决定标注输出写 mp4 还是逐帧存图）。
    is_video: bool,
    /// 收集所有 JSON 结果，最后一次性写入。
    json_results: Vec<Value>,
    /// 失败的输入（路径 → 错误信息），批量模式下单张失败不中断。
    failures: Vec<Value>,
    /// 标注输出落盘器（图片 / 视频 / 逐帧统一走这里，含编码线程）。
    saver: SaveResults,
}

impl Output {
    /// 创建输出器，并确保输出目录存在。
    pub fn new(config: &OutputConfig, verbose: bool, task: &str, is_video: bool) -> Result<Self> {
        let dir = Self::build_run_dir(&config.dir, task);
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir: dir.clone(),
            save_annotated: config.save_annotated,
            save_json: config.save_json,
            verbose,
            task: task.to_string(),
            is_video,
            json_results: Vec::new(),
            failures: Vec::new(),
            saver: SaveResults::new(dir, config.save_frames),
        })
    }

    /// 输出子目录：`<root>/<task>/<时间戳>`。
    fn build_run_dir(root: &str, task: &str) -> PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        PathBuf::from(root).join(task).join(ts.to_string())
    }

    /// 处理单个推理结果。
    pub fn handle(&mut self, result: &Results, image: &DynamicImage, meta: &SourceMeta) -> Result<()> {
        if self.verbose {
            println!(
                "{}: {} ({:.1} ms)",
                result.path,
                result.detection_summary(),
                result.speed.total()
            );
        }

        if self.save_json {
            self.json_results.push(self.result_to_json(result, meta));
        }

        if self.save_annotated {
            #[cfg(feature = "annotate")]
            {
                let annotated = annotate_image(image, result, None);
                self.saver.save(self.is_video, meta, &annotated)?;
            }
        }

        Ok(())
    }

    /// 记录一个失败输入（解码失败或推理失败）。
    pub fn record_failure(&mut self, path: &str, error: &str) {
        self.failures
            .push(json!({ "path": path, "error": error }));
    }

    /// 落盘 JSON 汇总（全部结果处理完后调用一次）。
    pub fn write_json(&self) -> Result<()> {
        if !self.save_json {
            return Ok(());
        }
        let payload = json!({ "results": &self.json_results, "failures": &self.failures });
        let path = self.dir.join("results.json");
        std::fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
        println!("JSON 已保存: {}", path.display());
        Ok(())
    }

    /// 收尾：落盘 JSON 汇总，并等待所有标注图 / 视频帧编码完成。
    pub fn finish(self) -> Result<()> {
        self.write_json()?;
        self.saver.finish()?;
        Ok(())
    }

    /// 把一个 `Results` 转成 JSON 值。
    fn result_to_json(&self, result: &Results, meta: &SourceMeta) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("path".into(), json!(result.path));
        obj.insert("frame_idx".into(), json!(meta.frame_idx));
        obj.insert("fps".into(), json!(meta.fps));
        obj.insert("task".into(), json!(self.task));
        obj.insert(
            "orig_shape".into(),
            json!([result.orig_shape.0, result.orig_shape.1]),
        );
        obj.insert(
            "inference_shape".into(),
            json!([result.inference_shape.0, result.inference_shape.1]),
        );
        obj.insert(
            "speed_ms".into(),
            json!({
                "preprocess": result.speed.preprocess,
                "inference": result.speed.inference,
                "postprocess": result.speed.postprocess,
                "total": result.speed.total(),
            }),
        );

        // 任务载荷：只写入当前任务实际填充的字段。
        if let Some(boxes) = &result.boxes {
            obj.insert("detections".into(), boxes_to_json(boxes, &result.names));
        }
        if let Some(masks) = &result.masks {
            obj.insert("masks".into(), masks_to_json(masks));
        }
        if let Some(kpts) = &result.keypoints {
            obj.insert("keypoints".into(), keypoints_to_json(kpts));
        }
        if let Some(obb) = &result.obb {
            obj.insert("obb".into(), obb_to_json(obb, &result.names));
        }
        if let Some(probs) = &result.probs {
            obj.insert("probs".into(), probs_to_json(probs, &result.names));
        }
        if let Some(sem) = &result.semantic_mask {
            obj.insert("semantic".into(), semantic_to_json(sem, &result.names));
        }
        if let Some(depth) = &result.depth {
            obj.insert("depth".into(), depth_to_json(depth));
        }

        Value::Object(obj)
    }
}

/// 类别 ID → 名称，未知 ID 回退为数字字符串。
fn name_for(names: &HashMap<usize, String>, class_id: usize) -> String {
    names
        .get(&class_id)
        .cloned()
        .unwrap_or_else(|| class_id.to_string())
}

/// 检测框（detect / segment / pose 的 `boxes`）：归一化 xyxy。
fn boxes_to_json(boxes: &Boxes, names: &HashMap<usize, String>) -> Value {
    let xyxyn = boxes.xyxyn();
    let conf = boxes.conf();
    let cls = boxes.cls();

    let items: Vec<Value> = (0..boxes.len())
        .map(|i| {
            let class_id = cls[i] as usize;
            json!({
                "name": name_for(names, class_id),
                "class": class_id,
                "confidence": conf[i],
                "box": {
                    "x1": xyxyn[[i, 0]],
                    "y1": xyxyn[[i, 1]],
                    "x2": xyxyn[[i, 2]],
                    "y2": xyxyn[[i, 3]],
                },
            })
        })
        .collect();

    json!(items)
}

/// 实例分割掩码（segment）：每个实例的二值面积占比 + 掩码分辨率。
/// 不落盘逐像素掩码（体积过大），完整掩码可视化见标注图。
fn masks_to_json(masks: &Masks) -> Value {
    let shape = masks.data.shape();
    let (h, w) = (shape[1], shape[2]);
    let total = (h * w) as f32;

    let areas: Vec<f32> = masks
        .data
        .outer_iter()
        .map(|m| m.iter().filter(|&&v| v > 0.5).count() as f32 / total)
        .collect();

    json!({
        "count": masks.len(),
        "shape": [h, w],
        "areas": areas,
    })
}

/// 姿态关键点（pose）：每人一组，`[x, y, conf]`（无置信度通道时 conf=1.0）。
fn keypoints_to_json(kpts: &Keypoints) -> Value {
    let xyn = kpts.xyn();
    let conf = kpts.conf();
    let (n, k) = (kpts.len(), xyn.shape()[1]);

    let persons: Vec<Value> = (0..n)
        .map(|i| {
            let pts: Vec<Value> = (0..k)
                .map(|j| {
                    let c = conf.as_ref().map_or(1.0, |c| c[[i, j]]);
                    json!([xyn[[i, j, 0]], xyn[[i, j, 1]], c])
                })
                .collect();
            Value::Array(pts)
        })
        .collect();

    json!({
        "count": n,
        "num_keypoints": k,
        "has_confidence": conf.is_some(),
        "keypoints": persons,
    })
}

/// 旋转目标框（obb）：归一化的 `cx/cy/w/h/rotation` + 四个角点。
fn obb_to_json(obb: &Obb, names: &HashMap<usize, String>) -> Value {
    let (h, w) = (obb.orig_shape.0 as f32, obb.orig_shape.1 as f32);
    let xywhr = obb.xywhr();
    let corners = obb.xyxyxyxy();
    let conf = obb.conf();
    let cls = obb.cls();

    let items: Vec<Value> = (0..obb.len())
        .map(|i| {
            let class_id = cls[i] as usize;
            let pts: Vec<Value> = (0..4)
                .map(|j| json!([corners[[i, j, 0]] / w, corners[[i, j, 1]] / h]))
                .collect();
            json!({
                "name": name_for(names, class_id),
                "class": class_id,
                "confidence": conf[i],
                "obb": {
                    "cx": xywhr[[i, 0]] / w,
                    "cy": xywhr[[i, 1]] / h,
                    "w": xywhr[[i, 2]] / w,
                    "h": xywhr[[i, 3]] / h,
                    "rotation": xywhr[[i, 4]],
                },
                "corners": pts,
            })
        })
        .collect();

    json!(items)
}

/// 分类概率（classify）：top-1 与 top-5。
fn probs_to_json(probs: &Probs, names: &HashMap<usize, String>) -> Value {
    let top5 = probs.top5();
    let top5conf = probs.top5conf();

    let items: Vec<Value> = top5
        .iter()
        .enumerate()
        .map(|(idx, &class_id)| {
            json!({
                "class": class_id,
                "name": name_for(names, class_id),
                "confidence": top5conf[idx],
            })
        })
        .collect();

    json!({
        "top1": items.first().cloned().unwrap_or(Value::Null),
        "top5": items,
    })
}

/// 语义分割（semantic）：出现的类别 + 各类像素占比。
fn semantic_to_json(sem: &SemanticMask, names: &HashMap<usize, String>) -> Value {
    let total = sem.data.len() as f32;
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for &v in sem.data.iter() {
        if v != SemanticMask::IGNORE {
            *counts.entry(v).or_insert(0) += 1;
        }
    }

    let mut ids: Vec<u16> = counts.keys().copied().collect();
    ids.sort_unstable();

    let mut classes = Vec::with_capacity(ids.len());
    let mut fractions = serde_json::Map::new();
    for id in ids {
        let class_id = usize::from(id);
        classes.push(json!({
            "class": class_id,
            "name": name_for(names, class_id),
        }));
        fractions.insert(id.to_string(), json!(counts[&id] as f32 / total));
    }

    json!({
        "shape": [sem.data.nrows(), sem.data.ncols()],
        "classes": classes,
        "pixel_fractions": Value::Object(fractions),
    })
}

/// 深度图（depth）：有效像素的 min/max + 分辨率（不落盘整张深度图）。
fn depth_to_json(depth: &DepthMap) -> Value {
    json!({
        "min_m": depth.min_depth(),
        "max_m": depth.max_depth(),
        "shape": [depth.data.nrows(), depth.data.ncols()],
    })
}
