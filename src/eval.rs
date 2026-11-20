//! 精度评估（阶段 4）：按模型任务分派。
//!
//! - detect：COCO 标注 + 框 mAP（101 点插值）。
//! - classify：简单标签 JSON + top1 / top5 准确率。
//! - pose：COCO 关键点 + OKS 的 mAP（对标 `model.val()`）+ bbox 归一化 PCK@0.5。
//!
//! 用法：`cargo run -- eval config/eval.toml`，产出 `eval_report.json`。
//! 预测与真值按**类别名称**对齐（不依赖类别 ID 排序）。

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Value, json};
use ultralytics_inference::{Results, SemanticMask, YOLOModel};

use crate::config::{DatasetConfig, EvalConfig};
use crate::error::{AppError, Result};

/// mAP@0.5:0.95 用的 10 个 IoU 阈值。
const IOU_THRESHOLDS: [f32; 10] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95];

/// 运行精度评估，按模型任务分派，报告写入 `<eval.dir>/eval_report.json`。
pub fn run_eval(config: &EvalConfig) -> Result<()> {
    let inference_config = config.inference.to_inference_config()?;
    let mut model = YOLOModel::load_with_config(&config.model.path, inference_config)?;
    let task = model.task().as_str().to_string();
    println!(
        "评估模型: {} | 任务: {} | 设备: {}",
        config.model.path,
        task,
        model.execution_provider()
    );

    match task.as_str() {
        "classify" => run_classify(&mut model, config),
        "pose" => run_pose(&mut model, config),
        "segment" => run_segment(&mut model, config),
        "semantic" => run_semantic(&mut model, config),
        "obb" => run_obb(&mut model, config),
        _ => run_detect(&mut model, config, &task),
    }
}

/// detect：COCO 标注 + 框 mAP。
fn run_detect(model: &mut YOLOModel, config: &EvalConfig, task: &str) -> Result<()> {
    let coco = CocoDataset::load(&config.dataset)?;
    println!(
        "评估集: {} 张图, {} 个 GT 框",
        coco.images.len(),
        coco.gts.len()
    );

    let images_dir = PathBuf::from(&config.dataset.images);
    let mut detections = Vec::new();
    let mut skipped = 0usize;
    for img in &coco.images {
        let path = images_dir.join(&img.file_name);
        if !path.is_file() {
            eprintln!("跳过（图片不存在）: {}", path.display());
            skipped += 1;
            continue;
        }
        let results = model.predict(&path)?;
        detections.extend(collect_detections(&results, img.id));
    }
    if skipped == coco.images.len() {
        return Err(AppError::Config(format!(
            "评估集图片全部缺失，请检查 dataset.images 目录: {}",
            config.dataset.images
        )));
    }
    println!(
        "推理完成: {} 个预测框（跳过 {} 张缺失图）",
        detections.len(),
        skipped
    );

    let report = json!({
        "model": config.model.path,
        "task": task,
        "dataset": {
            "images": config.dataset.images,
            "annotations": config.dataset.annotations,
        },
        "num_images": coco.images.len() - skipped,
        "num_gt_boxes": coco.gts.len(),
        "num_predictions": detections.len(),
        "eval": evaluate(&coco.gts, &detections),
    });

    write_report(&config.eval.dir, &report)
}

/// classify：`annotations` 为简单标签 JSON，计算 top1 / top5 准确率。
fn run_classify(model: &mut YOLOModel, config: &EvalConfig) -> Result<()> {
    let text = std::fs::read_to_string(&config.dataset.annotations).map_err(|source| {
        AppError::ConfigIo {
            path: config.dataset.annotations.clone(),
            source,
        }
    })?;
    let labels: ClassifyLabels = serde_json::from_str(&text)?;
    let images_dir = PathBuf::from(&config.dataset.images);

    let mut correct1 = 0usize;
    let mut correct5 = 0usize;
    let mut total = 0usize;
    let mut skipped = 0usize;
    let mut per_image = Vec::new();

    for item in &labels.images {
        let path = images_dir.join(&item.file);
        if !path.is_file() {
            eprintln!("跳过（图片不存在）: {}", path.display());
            skipped += 1;
            continue;
        }
        let results = model.predict(&path)?;
        for r in results {
            let Some(probs) = &r.probs else { continue };
            let top5: Vec<String> = probs.top5().iter().map(|&i| class_name(&r, i)).collect();
            let (hit1, hit5) = topk_hits(&top5, &item.class);
            if hit1 {
                correct1 += 1;
            }
            if hit5 {
                correct5 += 1;
            }
            total += 1;
            per_image.push(json!({
                "file": item.file,
                "true": item.class,
                "top1": top5.first().cloned().unwrap_or_default(),
                "top5": top5,
                "top1_correct": hit1,
                "top5_correct": hit5,
            }));
        }
    }

    let report = json!({
        "model": config.model.path,
        "task": "classify",
        "dataset": {
            "images": config.dataset.images,
            "annotations": config.dataset.annotations,
        },
        "num_images": total,
        "num_skipped": skipped,
        "eval": {
            "top1_accuracy": if total > 0 { correct1 as f64 / total as f64 } else { 0.0 },
            "top5_accuracy": if total > 0 { correct5 as f64 / total as f64 } else { 0.0 },
            "per_image": per_image,
        },
    });

    write_report(&config.eval.dir, &report)
}

/// 写评估报告 JSON。
fn write_report(dir: &str, report: &Value) -> Result<()> {
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("eval_report.json");
    std::fs::write(&path, serde_json::to_string_pretty(report)?)?;
    println!("评估报告已保存: {}", path.display());
    Ok(())
}

/// 类别索引 → 名称（未知回退为数字字符串）。
fn class_name(r: &Results, idx: usize) -> String {
    r.names
        .get(&idx)
        .cloned()
        .unwrap_or_else(|| idx.to_string())
}

/// top-k 命中：`(top1 是否命中, top5 是否命中)`。
fn topk_hits(top5: &[String], truth: &str) -> (bool, bool) {
    let top1 = top5.first().map(String::as_str).unwrap_or("");
    (top1 == truth, top5.iter().any(|t| t == truth))
}

/// 从一次推理的 `Results` 中收集检测框（像素坐标 xyxy + 类别名称）。
fn collect_detections(results: &[Results], image_id: i64) -> Vec<Detection> {
    let mut dets = Vec::new();
    for r in results {
        let Some(boxes) = &r.boxes else { continue };
        let xyxy = boxes.xyxy();
        let conf = boxes.conf();
        let cls = boxes.cls();
        for i in 0..boxes.len() {
            let class_id = cls[i] as usize;
            let class = r
                .names
                .get(&class_id)
                .cloned()
                .unwrap_or_else(|| class_id.to_string());
            dets.push(Detection {
                image_id,
                class,
                conf: conf[i],
                bbox: [xyxy[[i, 0]], xyxy[[i, 1]], xyxy[[i, 2]], xyxy[[i, 3]]],
            });
        }
    }
    dets
}

/// 汇总 mAP：按类别（真值出现的类别）计算 AP，再取均值。
fn evaluate(gts: &[Gt], dets: &[Detection]) -> Value {
    let mut dets_by_class: HashMap<String, Vec<Detection>> = HashMap::new();
    for d in dets {
        dets_by_class.entry(d.class.clone()).or_default().push(d.clone());
    }
    let mut gts_by_class: HashMap<String, Vec<Gt>> = HashMap::new();
    for g in gts {
        gts_by_class.entry(g.class.clone()).or_default().push(g.clone());
    }

    let mut classes: Vec<String> = gts_by_class.keys().cloned().collect();
    classes.sort();

    let mut per_class = Vec::new();
    let mut sum_ap50 = 0.0f32;
    let mut sum_ap50_95 = 0.0f32;

    for cls in &classes {
        let cd = dets_by_class.get(cls).map(Vec::as_slice).unwrap_or(&[]);
        let cg = &gts_by_class[cls];
        let r = evaluate_class(cd, cg);
        per_class.push(json!({
            "class": cls,
            "n_gt": r.n_gt,
            "ap50": r.ap50,
            "ap50_95": r.ap50_95,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
        }));
        sum_ap50 += r.ap50;
        sum_ap50_95 += r.ap50_95;
    }

    let n = classes.len();
    json!({
        "num_classes": n,
        "mAP50": if n > 0 { sum_ap50 / n as f32 } else { 0.0 },
        "mAP50_95": if n > 0 { sum_ap50_95 / n as f32 } else { 0.0 },
        "per_class": per_class,
    })
}

/// 单个类别的 AP 汇总。
struct ClassReport {
    n_gt: usize,
    ap50: f32,
    ap50_95: f32,
    precision: f32,
    recall: f32,
    f1: f32,
}

/// 计算某个类别在全部 IoU 阈值下的 AP，以及在 IoU=0.5 处最大 F1 点的 P/R/F1。
fn evaluate_class(dets: &[Detection], gts: &[Gt]) -> ClassReport {
    let mut aps = [0.0f32; IOU_THRESHOLDS.len()];
    let mut prf = (0.0f32, 0.0f32, 0.0f32);
    for (i, &t) in IOU_THRESHOLDS.iter().enumerate() {
        let (sorted, n_gt) = match_at_threshold(dets, gts, t);
        let (ap, p, r, f1) = compute_ap(&sorted, n_gt);
        aps[i] = ap;
        if t == 0.5 {
            prf = (p, r, f1);
        }
    }
    ClassReport {
        n_gt: gts.len(),
        ap50: aps[0],
        ap50_95: aps.iter().sum::<f32>() / aps.len() as f32,
        precision: prf.0,
        recall: prf.1,
        f1: prf.2,
    }
}

/// 在某个 IoU 阈值下做贪心匹配，返回按置信度降序的 `(conf, tp)` 序列与 GT 数。
///
/// 匹配按「图像内」进行：某张图的预测只能匹配同一张图、同一类别的真值。
fn match_at_threshold(dets: &[Detection], gts: &[Gt], threshold: f32) -> (Vec<(f32, bool)>, usize) {
    let mut dets_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, d) in dets.iter().enumerate() {
        dets_by_image.entry(d.image_id).or_default().push(i);
    }
    let mut gts_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, g) in gts.iter().enumerate() {
        gts_by_image.entry(g.image_id).or_default().push(i);
    }

    let mut matched = vec![false; gts.len()];
    let mut out: Vec<(f32, bool)> = Vec::with_capacity(dets.len());

    for (image_id, det_indices) in &dets_by_image {
        let mut sorted = det_indices.clone();
        sorted.sort_by(|&a, &b| {
            dets[b]
                .conf
                .partial_cmp(&dets[a].conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let gt_indices = gts_by_image.get(image_id);
        for &di in &sorted {
            let d = &dets[di];
            let mut best: Option<(usize, f32)> = None;
            if let Some(gt_indices) = gt_indices {
                for &gi in gt_indices {
                    if matched[gi] {
                        continue;
                    }
                    let iou = box_iou(&d.bbox, &gts[gi].bbox);
                    if iou > best.map_or(0.0, |(_, b)| b) {
                        best = Some((gi, iou));
                    }
                }
            }
            match best {
                Some((gi, iou)) if iou >= threshold => {
                    matched[gi] = true;
                    out.push((d.conf, true));
                }
                _ => out.push((d.conf, false)),
            }
        }
    }

    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (out, gts.len())
}

/// 由 `(conf, tp)` 降序序列计算 AP（101 点插值）与最大 F1 点的 P/R/F1。
fn compute_ap(sorted: &[(f32, bool)], n_gt: usize) -> (f32, f32, f32, f32) {
    if n_gt == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut precisions = Vec::with_capacity(sorted.len());
    let mut recalls = Vec::with_capacity(sorted.len());
    for &(_, is_tp) in sorted {
        if is_tp {
            tp += 1;
        } else {
            fp += 1;
        }
        precisions.push(tp as f32 / (tp + fp) as f32);
        recalls.push(tp as f32 / n_gt as f32);
    }

    // 与 ultralytics `compute_ap`（method="interp"）对齐：精度包络 + 101 点线性插值 + 梯形积分。
    let ap = ap_linear_interp(&recalls, &precisions);

    // 最大 F1 点（原始 PR 曲线）。
    let mut best = (0.0f32, 0.0f32, 0.0f32);
    for (&prec, &rec) in precisions.iter().zip(&recalls) {
        let f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };
        if f1 > best.0 {
            best = (f1, prec, rec);
        }
    }

    (ap, best.1, best.2, best.0)
}

/// 与 ultralytics `compute_ap`（`method="interp"`）一致的 AP：精度包络 + 101 点线性插值 + 梯形积分。
///
/// ```python
/// mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
/// mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
/// mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision envelope
/// x = np.linspace(0, 1, 101)
/// ap = np.trapezoid(np.interp(x, mrec, mpre), x)
/// ```
fn ap_linear_interp(recalls: &[f32], precisions: &[f32]) -> f32 {
    let n = recalls.len();
    let last_recall = if n > 0 { recalls[n - 1] } else { 1.0 };
    // mrec = [0] + recall + [last_recall, 1]
    let mut mrec = Vec::with_capacity(n + 3);
    mrec.push(0.0f32);
    mrec.extend_from_slice(recalls);
    mrec.push(last_recall);
    mrec.push(1.0f32);
    // mpre = [1] + precision + [0, 0]
    let mut mpre = Vec::with_capacity(n + 3);
    mpre.push(1.0f32);
    mpre.extend_from_slice(precisions);
    mpre.push(0.0f32);
    mpre.push(0.0f32);

    // 精度包络：从右往左取累积最大值，使 mpre 随 recall 单调非增。
    for i in (0..mpre.len() - 1).rev() {
        if mpre[i] < mpre[i + 1] {
            mpre[i] = mpre[i + 1];
        }
    }

    // 101 点线性插值 + 梯形积分（np.trapezoid(y, x)，x = linspace(0, 1, 101)）。
    let step = 1.0f32 / 100.0;
    let mut ap = 0.0f32;
    for i in 0..=100 {
        let x = i as f32 * step;
        let y = interp_linear(x, &mrec, &mpre);
        let w = if i == 0 || i == 100 { 0.5f32 } else { 1.0f32 };
        ap += w * y;
    }
    ap * step
}

/// `np.interp` 的一维线性插值（`xp` 单调非减，可含重复值；`x` 越界时取端点值）。
///
/// 复刻 numpy 对重复 xp 的语义：`idx = np.searchsorted(xp, x, side="right")`（首个 `xp[idx] > x`），
/// 在 `[idx-1, idx]` 之间线性插值；重复值处取「最右」的那个 y。
fn interp_linear(x: f32, xp: &[f32], fp: &[f32]) -> f32 {
    let n = xp.len();
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    // 二分查找首个 xp[mid] > x 的索引（searchsorted side="right"）。
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return fp[0];
    }
    if lo == n {
        return fp[n - 1];
    }
    let (x0, x1) = (xp[lo - 1], xp[lo]);
    let (y0, y1) = (fp[lo - 1], fp[lo]);
    if x1 == x0 {
        y1
    } else {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

/// 两个 xyxy 框的 IoU。
fn box_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// 一个检测框（像素坐标 xyxy）。
#[derive(Debug, Clone)]
struct Detection {
    image_id: i64,
    class: String,
    conf: f32,
    bbox: [f32; 4],
}

/// 一个真值框（像素坐标 xyxy）。
#[derive(Debug, Clone)]
struct Gt {
    image_id: i64,
    class: String,
    bbox: [f32; 4],
}

/// COCO 标注 JSON 的子集（只取评估需要的字段）。
#[derive(Debug, Deserialize)]
struct CocoJson {
    #[serde(default)]
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnn>,
    #[serde(default)]
    categories: Vec<CocoCategory>,
}

#[derive(Debug, Deserialize)]
struct CocoImage {
    id: i64,
    file_name: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
}

#[derive(Debug, Deserialize)]
struct CocoAnn {
    image_id: i64,
    category_id: i64,
    bbox: [f64; 4],
    #[serde(default)]
    iscrowd: u8,
    #[serde(default)]
    area: f64,
    #[serde(default)]
    keypoints: Vec<f64>,
    #[serde(default)]
    segmentation: Option<Segmentation>,
}

/// COCO `segmentation` 字段：多边形（像素坐标）或 RLE。
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Segmentation {
    Rle(SegRle),
    Polygons(Vec<Vec<f64>>),
}

/// COCO RLE：`{"size": [h, w], "counts": ...}`。
#[derive(Debug, Deserialize)]
struct SegRle {
    #[serde(default)]
    size: Option<[u32; 2]>,
    counts: SegCounts,
}

/// RLE 的 `counts`：未压缩（整数游程）或压缩（字符串，暂不支持）。
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SegCounts {
    /// 压缩 RLE（COCO-stuff / 全景）仅被识别用于跳过，内容不解析。
    #[allow(dead_code)]
    Compressed(String),
    Uncompressed(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct CocoCategory {
    id: i64,
    name: String,
}

/// classify 标签 JSON：`{"images": [{"file", "class"}]}`。
#[derive(Debug, Deserialize)]
struct ClassifyLabels {
    #[serde(default)]
    images: Vec<ClassifyItem>,
}

#[derive(Debug, Deserialize)]
struct ClassifyItem {
    file: String,
    class: String,
}

/// 加载后的数据集。
struct CocoDataset {
    images: Vec<CocoImage>,
    gts: Vec<Gt>,
}

impl CocoDataset {
    /// 从配置加载 COCO JSON，把 `bbox`（xywh）转成 xyxy 像素坐标。
    fn load(dataset: &DatasetConfig) -> Result<Self> {
        let coco = read_coco(dataset)?;
        let cat_name = coco_category_names(&coco);

        let gts = coco
            .annotations
            .iter()
            .filter(|ann| ann.iscrowd == 0) // 与 ultralytics convert_coco 一致：跳过 iscrowd 标注
            .filter_map(|ann| {
                let name = cat_name.get(&ann.category_id)?;
                let [x, y, w, h] = ann.bbox;
                Some(Gt {
                    image_id: ann.image_id,
                    class: name.clone(),
                    bbox: [x as f32, y as f32, (x + w) as f32, (y + h) as f32],
                })
            })
            .collect();

        Ok(Self {
            images: coco.images,
            gts,
        })
    }
}

/// 读取并解析 COCO 标注 JSON。
fn read_coco(dataset: &DatasetConfig) -> Result<CocoJson> {
    let text = std::fs::read_to_string(&dataset.annotations).map_err(|source| {
        AppError::ConfigIo {
            path: dataset.annotations.clone(),
            source,
        }
    })?;
    Ok(serde_json::from_str(&text)?)
}

/// 类别 ID → 名称。
fn coco_category_names(coco: &CocoJson) -> HashMap<i64, String> {
    coco.categories
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect()
}

/// COCO 人体 17 关键点的 OKS 归一化系数（标准值 / 10，即 `σ_i`）。
///
/// 与 pycocotools 一致：`vars = (2 * σ_i)^2`，`e = d^2 / vars / area / 2`。
const PERSON_SIGMAS: [f32; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107,
    0.107, 0.087, 0.087, 0.089, 0.089,
];

/// pose：COCO 关键点标注 + OKS mAP + PCK@0.5。
fn run_pose(model: &mut YOLOModel, config: &EvalConfig) -> Result<()> {
    let (images, gts) = load_pose_gts(&config.dataset)?;
    println!("评估集: {} 张图, {} 个 GT 实例", images.len(), gts.len());

    let images_dir = PathBuf::from(&config.dataset.images);
    let mut dets = Vec::new();
    let mut skipped = 0usize;
    for img in &images {
        let path = images_dir.join(&img.file_name);
        if !path.is_file() {
            eprintln!("跳过（图片不存在）: {}", path.display());
            skipped += 1;
            continue;
        }
        let results = model.predict(&path)?;
        dets.extend(collect_pose_detections(&results, img.id));
    }
    if skipped == images.len() {
        return Err(AppError::Config(format!(
            "评估集图片全部缺失，请检查 dataset.images 目录: {}",
            config.dataset.images
        )));
    }
    println!(
        "推理完成: {} 个预测实例（跳过 {} 张缺失图）",
        dets.len(),
        skipped
    );

    let report = json!({
        "model": config.model.path,
        "task": "pose",
        "dataset": {
            "images": config.dataset.images,
            "annotations": config.dataset.annotations,
        },
        "num_images": images.len() - skipped,
        "num_gt": gts.len(),
        "num_predictions": dets.len(),
        "eval": evaluate_pose(&gts, &dets),
    });

    write_report(&config.eval.dir, &report)
}

/// 从一次推理的 `Results` 中收集姿态实例（像素 xyxy 框 + 像素关键点）。
fn collect_pose_detections(results: &[Results], image_id: i64) -> Vec<PoseDetection> {
    let mut dets = Vec::new();
    for r in results {
        let (Some(boxes), Some(kpts)) = (&r.boxes, &r.keypoints) else {
            continue;
        };
        let conf = boxes.conf();
        let cls = boxes.cls();
        let xy = kpts.xy();
        let n_kpts = xy.shape()[1];
        let n = boxes.len().min(kpts.len());
        for i in 0..n {
            let class_id = cls[i] as usize;
            let class = r
                .names
                .get(&class_id)
                .cloned()
                .unwrap_or_else(|| class_id.to_string());
            let mut keypoints = Vec::with_capacity(n_kpts * 2);
            for j in 0..n_kpts {
                keypoints.push(xy[[i, j, 0]]);
                keypoints.push(xy[[i, j, 1]]);
            }
            dets.push(PoseDetection {
                image_id,
                class,
                conf: conf[i],
                keypoints,
            });
        }
    }
    dets
}

/// 加载 pose 数据集：`Vec<CocoImage>` + `Vec<PoseGt>`（关键点 + 面积）。
fn load_pose_gts(dataset: &DatasetConfig) -> Result<(Vec<CocoImage>, Vec<PoseGt>)> {
    let coco = read_coco(dataset)?;
    let cat_name = coco_category_names(&coco);

    let gts = coco
        .annotations
        .iter()
        .filter_map(|ann| {
            let name = cat_name.get(&ann.category_id)?;
            let [_, _, w, h] = ann.bbox;
            let area = if ann.area > 0.0 {
                ann.area as f32
            } else {
                (w * h) as f32
            };
            Some(PoseGt {
                image_id: ann.image_id,
                class: name.clone(),
                area,
                keypoints: ann.keypoints.iter().map(|&v| v as f32).collect(),
            })
        })
        .collect();

    Ok((coco.images, gts))
}

/// pose 汇总：按类别（真值出现的类别）计算 OKS mAP + PCK@0.5，再取均值。
fn evaluate_pose(gts: &[PoseGt], dets: &[PoseDetection]) -> Value {
    let mut dets_by_class: HashMap<String, Vec<PoseDetection>> = HashMap::new();
    for d in dets {
        dets_by_class.entry(d.class.clone()).or_default().push(d.clone());
    }
    let mut gts_by_class: HashMap<String, Vec<PoseGt>> = HashMap::new();
    for g in gts {
        gts_by_class.entry(g.class.clone()).or_default().push(g.clone());
    }

    let mut classes: Vec<String> = gts_by_class.keys().cloned().collect();
    classes.sort();

    let mut per_class = Vec::new();
    let mut sum_ap50 = 0.0f32;
    let mut sum_ap50_95 = 0.0f32;
    let mut sum_pck = 0.0f32;

    for cls in &classes {
        let cd = dets_by_class.get(cls).map(Vec::as_slice).unwrap_or(&[]);
        let cg = &gts_by_class[cls];
        let r = evaluate_pose_class(cd, cg);
        let pck = compute_pck(cd, cg);
        per_class.push(json!({
            "class": cls,
            "n_gt": r.n_gt,
            "ap50": r.ap50,
            "ap50_95": r.ap50_95,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "pck50": pck,
        }));
        sum_ap50 += r.ap50;
        sum_ap50_95 += r.ap50_95;
        sum_pck += pck;
    }

    let n = classes.len();
    json!({
        "num_classes": n,
        "mAP50": if n > 0 { sum_ap50 / n as f32 } else { 0.0 },
        "mAP50_95": if n > 0 { sum_ap50_95 / n as f32 } else { 0.0 },
        "pck50": if n > 0 { sum_pck / n as f32 } else { 0.0 },
        "per_class": per_class,
    })
}

/// 单个类别的 OKS AP 汇总（OKS 阈值网格与框 IoU 相同：0.5..0.95）。
fn evaluate_pose_class(dets: &[PoseDetection], gts: &[PoseGt]) -> ClassReport {
    let mut aps = [0.0f32; IOU_THRESHOLDS.len()];
    let mut prf = (0.0f32, 0.0f32, 0.0f32);
    for (i, &t) in IOU_THRESHOLDS.iter().enumerate() {
        let (sorted, _) = match_pose(dets, gts, t);
        let (ap, p, r, f1) = compute_ap(&sorted, gts.len());
        aps[i] = ap;
        if t == 0.5 {
            prf = (p, r, f1);
        }
    }
    ClassReport {
        n_gt: gts.len(),
        ap50: aps[0],
        ap50_95: aps.iter().sum::<f32>() / aps.len() as f32,
        precision: prf.0,
        recall: prf.1,
        f1: prf.2,
    }
}

/// 在某个 OKS 阈值下做贪心匹配（图像内、同类）。
///
/// 返回按置信度降序的 `(conf, tp)` 序列与匹配上的 `(预测索引, 真值索引)` 对。
fn match_pose(
    dets: &[PoseDetection],
    gts: &[PoseGt],
    threshold: f32,
) -> (Vec<(f32, bool)>, Vec<(usize, usize)>) {
    let mut dets_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, d) in dets.iter().enumerate() {
        dets_by_image.entry(d.image_id).or_default().push(i);
    }
    let mut gts_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, g) in gts.iter().enumerate() {
        gts_by_image.entry(g.image_id).or_default().push(i);
    }

    let mut matched_gt = vec![false; gts.len()];
    let mut pairs = Vec::new();
    let mut out: Vec<(f32, bool)> = Vec::with_capacity(dets.len());

    for (image_id, det_indices) in &dets_by_image {
        let mut sorted = det_indices.clone();
        sorted.sort_by(|&a, &b| {
            dets[b]
                .conf
                .partial_cmp(&dets[a].conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let gt_indices = gts_by_image.get(image_id);
        for &di in &sorted {
            let d = &dets[di];
            let mut best: Option<(usize, f32)> = None;
            if let Some(gt_indices) = gt_indices {
                for &gi in gt_indices {
                    if matched_gt[gi] {
                        continue;
                    }
                    let s = oks(&d.keypoints, &gts[gi].keypoints, gts[gi].area);
                    if s > best.map_or(0.0, |(_, b)| b) {
                        best = Some((gi, s));
                    }
                }
            }
            match best {
                Some((gi, s)) if s >= threshold => {
                    matched_gt[gi] = true;
                    pairs.push((di, gi));
                    out.push((d.conf, true));
                }
                _ => out.push((d.conf, false)),
            }
        }
    }

    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (out, pairs)
}

/// 预测关键点（像素 `[x,y]×K`）与真值关键点（像素 `[x,y,v]×K`）之间的 OKS。
///
/// 只对真值中「已标注」（`v > 0`）的关键点求均值；面积用 `area.max(ε)` 防除零。
fn oks(dt_kpts: &[f32], gt_kpts: &[f32], area: f32) -> f32 {
    let n = (gt_kpts.len() / 3)
        .min(dt_kpts.len() / 2)
        .min(PERSON_SIGMAS.len());
    let area = area.max(1e-6);
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        let v = gt_kpts[i * 3 + 2];
        if v <= 0.0 {
            continue;
        }
        let dx = dt_kpts[i * 2] - gt_kpts[i * 3];
        let dy = dt_kpts[i * 2 + 1] - gt_kpts[i * 3 + 1];
        let var = (2.0 * PERSON_SIGMAS[i]).powi(2);
        let e = (dx * dx + dy * dy) / var / area / 2.0;
        sum += (-e).exp();
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// PCK@0.5：在 OKS=0.5 贪心匹配上，统计「预测关键点与真值距离 ≤ 0.5·√面积」的比例。
///
/// 归一化基准用目标尺度 `√(bbox 面积)`（bbox 归一化 PCK，非 head 归一化的 PCKh）。
fn compute_pck(dets: &[PoseDetection], gts: &[PoseGt]) -> f32 {
    let (_, pairs) = match_pose(dets, gts, 0.5);
    let mut correct = 0usize;
    let mut total = 0usize;
    for &(di, gi) in &pairs {
        let dt = &dets[di].keypoints;
        let gt = &gts[gi].keypoints;
        let scale = gts[gi].area.sqrt();
        let n = (gt.len() / 3).min(dt.len() / 2);
        for i in 0..n {
            if gt[i * 3 + 2] <= 0.0 {
                continue;
            }
            let dx = dt[i * 2] - gt[i * 3];
            let dy = dt[i * 2 + 1] - gt[i * 3 + 1];
            total += 1;
            if (dx * dx + dy * dy).sqrt() <= 0.5 * scale {
                correct += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    }
}

/// 一个姿态预测实例（像素关键点；尺度用真值面积归一化）。
#[derive(Debug, Clone)]
struct PoseDetection {
    image_id: i64,
    class: String,
    conf: f32,
    keypoints: Vec<f32>,
}

/// 一个姿态真值实例（面积 + 关键点 `[x,y,v]×K`）。
#[derive(Debug, Clone)]
struct PoseGt {
    image_id: i64,
    class: String,
    area: f32,
    keypoints: Vec<f32>,
}

/// segment：COCO 分割标注 + 掩码 IoU 下的 mAP。
fn run_segment(model: &mut YOLOModel, config: &EvalConfig) -> Result<()> {
    let (images, gts) = load_segment_gts(&config.dataset)?;
    println!("评估集: {} 张图, {} 个 GT 实例", images.len(), gts.len());

    let images_dir = PathBuf::from(&config.dataset.images);
    let mut dets = Vec::new();
    let mut skipped = 0usize;
    for img in &images {
        let path = images_dir.join(&img.file_name);
        if !path.is_file() {
            eprintln!("跳过（图片不存在）: {}", path.display());
            skipped += 1;
            continue;
        }
        let results = model.predict(&path)?;
        dets.extend(collect_segment_detections(&results, img.id));
    }
    if skipped == images.len() {
        return Err(AppError::Config(format!(
            "评估集图片全部缺失，请检查 dataset.images 目录: {}",
            config.dataset.images
        )));
    }
    println!(
        "推理完成: {} 个预测实例（跳过 {} 张缺失图）",
        dets.len(),
        skipped
    );

    let report = json!({
        "model": config.model.path,
        "task": "segment",
        "dataset": {
            "images": config.dataset.images,
            "annotations": config.dataset.annotations,
        },
        "num_images": images.len() - skipped,
        "num_gt": gts.len(),
        "num_predictions": dets.len(),
        "eval": evaluate_segment(&gts, &dets),
    });

    write_report(&config.eval.dir, &report)
}

/// 从一次推理的 `Results` 中收集分割实例（掩码二值化到 `> 0.5`）。
fn collect_segment_detections(results: &[Results], image_id: i64) -> Vec<SegDetection> {
    let mut dets = Vec::new();
    for r in results {
        let (Some(boxes), Some(masks)) = (&r.boxes, &r.masks) else {
            continue;
        };
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
            let mut mask = Vec::with_capacity(mask_view.len());
            for &v in mask_view.iter() {
                mask.push(u8::from(v > 0.5));
            }
            dets.push(SegDetection {
                image_id,
                class,
                conf: conf[i],
                mask,
            });
        }
    }
    dets
}

/// 加载 segment 数据集：把 `segmentation`（多边形 / RLE）光栅化为二值掩码。
fn load_segment_gts(dataset: &DatasetConfig) -> Result<(Vec<CocoImage>, Vec<SegGt>)> {
    let coco = read_coco(dataset)?;
    let cat_name = coco_category_names(&coco);
    let dims: HashMap<i64, (usize, usize)> = coco
        .images
        .iter()
        .filter(|img| img.width > 0 && img.height > 0)
        .map(|img| (img.id, (img.height as usize, img.width as usize)))
        .collect();

    let mut gts = Vec::new();
    for ann in &coco.annotations {
        let Some(name) = cat_name.get(&ann.category_id) else {
            continue;
        };
        let Some(seg) = &ann.segmentation else {
            continue;
        };
        let (h, w) = match seg {
            Segmentation::Rle(rle) => match rle.size {
                Some([h, w]) => (h as usize, w as usize),
                None => {
                    let Some(&d) = dims.get(&ann.image_id) else {
                        continue;
                    };
                    d
                }
            },
            Segmentation::Polygons(_) => {
                let Some(&d) = dims.get(&ann.image_id) else {
                    continue;
                };
                d
            }
        };
        let Some(mask) = seg_to_mask(seg, h, w) else {
            continue;
        };
        gts.push(SegGt {
            image_id: ann.image_id,
            class: name.clone(),
            mask,
        });
    }

    Ok((coco.images, gts))
}

/// segment 汇总：按类别计算掩码 IoU 下的 mAP。
fn evaluate_segment(gts: &[SegGt], dets: &[SegDetection]) -> Value {
    let mut dets_by_class: HashMap<String, Vec<SegDetection>> = HashMap::new();
    for d in dets {
        dets_by_class.entry(d.class.clone()).or_default().push(d.clone());
    }
    let mut gts_by_class: HashMap<String, Vec<SegGt>> = HashMap::new();
    for g in gts {
        gts_by_class.entry(g.class.clone()).or_default().push(g.clone());
    }

    let mut classes: Vec<String> = gts_by_class.keys().cloned().collect();
    classes.sort();

    let mut per_class = Vec::new();
    let mut sum_ap50 = 0.0f32;
    let mut sum_ap50_95 = 0.0f32;

    for cls in &classes {
        let cd = dets_by_class.get(cls).map(Vec::as_slice).unwrap_or(&[]);
        let cg = &gts_by_class[cls];
        let r = evaluate_segment_class(cd, cg);
        per_class.push(json!({
            "class": cls,
            "n_gt": r.n_gt,
            "ap50": r.ap50,
            "ap50_95": r.ap50_95,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
        }));
        sum_ap50 += r.ap50;
        sum_ap50_95 += r.ap50_95;
    }

    let n = classes.len();
    json!({
        "num_classes": n,
        "mAP50": if n > 0 { sum_ap50 / n as f32 } else { 0.0 },
        "mAP50_95": if n > 0 { sum_ap50_95 / n as f32 } else { 0.0 },
        "per_class": per_class,
    })
}

/// 单个类别的掩码 IoU AP 汇总（阈值网格同框 IoU：0.5..0.95）。
fn evaluate_segment_class(dets: &[SegDetection], gts: &[SegGt]) -> ClassReport {
    let mut aps = [0.0f32; IOU_THRESHOLDS.len()];
    let mut prf = (0.0f32, 0.0f32, 0.0f32);
    for (i, &t) in IOU_THRESHOLDS.iter().enumerate() {
        let (sorted, n_gt) = match_segment(dets, gts, t);
        let (ap, p, r, f1) = compute_ap(&sorted, n_gt);
        aps[i] = ap;
        if t == 0.5 {
            prf = (p, r, f1);
        }
    }
    ClassReport {
        n_gt: gts.len(),
        ap50: aps[0],
        ap50_95: aps.iter().sum::<f32>() / aps.len() as f32,
        precision: prf.0,
        recall: prf.1,
        f1: prf.2,
    }
}

/// 在某个掩码 IoU 阈值下做贪心匹配（图像内、同类）。
fn match_segment(
    dets: &[SegDetection],
    gts: &[SegGt],
    threshold: f32,
) -> (Vec<(f32, bool)>, usize) {
    let mut dets_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, d) in dets.iter().enumerate() {
        dets_by_image.entry(d.image_id).or_default().push(i);
    }
    let mut gts_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, g) in gts.iter().enumerate() {
        gts_by_image.entry(g.image_id).or_default().push(i);
    }

    let mut matched = vec![false; gts.len()];
    let mut out: Vec<(f32, bool)> = Vec::with_capacity(dets.len());

    for (image_id, det_indices) in &dets_by_image {
        let mut sorted = det_indices.clone();
        sorted.sort_by(|&a, &b| {
            dets[b]
                .conf
                .partial_cmp(&dets[a].conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let gt_indices = gts_by_image.get(image_id);
        for &di in &sorted {
            let d = &dets[di];
            let mut best: Option<(usize, f32)> = None;
            if let Some(gt_indices) = gt_indices {
                for &gi in gt_indices {
                    if matched[gi] {
                        continue;
                    }
                    let iou = mask_iou(&d.mask, &gts[gi].mask);
                    if iou > best.map_or(0.0, |(_, b)| b) {
                        best = Some((gi, iou));
                    }
                }
            }
            match best {
                Some((gi, iou)) if iou >= threshold => {
                    matched[gi] = true;
                    out.push((d.conf, true));
                }
                _ => out.push((d.conf, false)),
            }
        }
    }

    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (out, gts.len())
}

/// 把 COCO `segmentation` 光栅化为 `h×w` 二值掩码（1 = 前景）。
///
/// 压缩 RLE（COCO-stuff / 全景）暂不支持，返回 `None`。
fn seg_to_mask(seg: &Segmentation, h: usize, w: usize) -> Option<Vec<u8>> {
    let mut mask = vec![0u8; h * w];
    match seg {
        Segmentation::Polygons(polys) => {
            for poly in polys {
                fill_polygon(&mut mask, h, w, poly);
            }
            Some(mask)
        }
        Segmentation::Rle(rle) => match &rle.counts {
            SegCounts::Uncompressed(counts) => {
                decode_rle(&mut mask, counts);
                Some(mask)
            }
            SegCounts::Compressed(_) => None,
        },
    }
}

/// 未压缩 RLE：`counts` 交替表示 0/1 游程（从 0 开始）。
fn decode_rle(mask: &mut [u8], counts: &[u32]) {
    let mut idx = 0usize;
    let mut val = 0u8;
    for &run in counts {
        for _ in 0..run {
            if idx < mask.len() {
                mask[idx] = val;
            }
            idx += 1;
        }
        val ^= 1;
    }
}

/// 用偶奇规则把多边形填充进掩码（像素中心点判定）。
fn fill_polygon(mask: &mut [u8], h: usize, w: usize, poly: &[f64]) {
    let n = poly.len() / 2;
    if n < 3 {
        return;
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        min_x = min_x.min(poly[2 * i]);
        max_x = max_x.max(poly[2 * i]);
        min_y = min_y.min(poly[2 * i + 1]);
        max_y = max_y.max(poly[2 * i + 1]);
    }

    let x0 = min_x.floor().max(0.0) as usize;
    let x1 = max_x.ceil().min(w as f64) as usize;
    let y0 = min_y.floor().max(0.0) as usize;
    let y1 = max_y.ceil().min(h as f64) as usize;

    for y in y0..y1 {
        for x in x0..x1 {
            if point_in_polygon(x as f64 + 0.5, y as f64 + 0.5, poly) {
                mask[y * w + x] = 1;
            }
        }
    }
}

/// 射线法判定点是否在多边形内（偶奇规则）。
fn point_in_polygon(px: f64, py: f64, poly: &[f64]) -> bool {
    let n = poly.len() / 2;
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[2 * i], poly[2 * i + 1]);
        let (xj, yj) = (poly[2 * j], poly[2 * j + 1]);
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// 两个二值掩码的 IoU（长度不一致视为 0）。
fn mask_iou(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut inter = 0u64;
    let mut union = 0u64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x == 1 && y == 1 {
            inter += 1;
        }
        if x == 1 || y == 1 {
            union += 1;
        }
    }
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

/// 一个分割预测实例（二值掩码）。
#[derive(Debug, Clone)]
struct SegDetection {
    image_id: i64,
    class: String,
    conf: f32,
    mask: Vec<u8>,
}

/// 一个分割真值实例（二值掩码）。
#[derive(Debug, Clone)]
struct SegGt {
    image_id: i64,
    class: String,
    mask: Vec<u8>,
}

/// semantic：像素级标签图（PNG）+ 逐类像素 IoU 的 mIoU。
fn run_semantic(model: &mut YOLOModel, config: &EvalConfig) -> Result<()> {
    let labels = load_semantic_labels(&config.dataset)?;
    let images_dir = PathBuf::from(&config.dataset.images);

    let mut acc: HashMap<u16, (u64, u64)> = HashMap::new();
    let mut names: HashMap<usize, String> = HashMap::new();
    let mut total = 0usize;
    let mut skipped = 0usize;

    for item in &labels.images {
        let path = images_dir.join(&item.file);
        let label_path = images_dir.join(&item.label);
        if !path.is_file() {
            eprintln!("跳过（图片不存在）: {}", path.display());
            skipped += 1;
            continue;
        }
        if !label_path.is_file() {
            eprintln!("跳过（标签图不存在）: {}", label_path.display());
            skipped += 1;
            continue;
        }
        let (gw, gh, gt) = load_semantic_label(&label_path)?;
        let results = model.predict(&path)?;
        let mut handled = false;
        for r in results {
            let Some(sem) = &r.semantic_mask else { continue };
            for (&id, name) in r.names.iter() {
                names.entry(id).or_insert_with(|| name.clone());
            }
            let (ph, pw) = (sem.data.nrows(), sem.data.ncols());
            if ph != gh as usize || pw != gw as usize {
                eprintln!(
                    "跳过（尺寸不一致）: {} 预测 {}x{} vs 标签 {}x{}",
                    item.file, pw, ph, gw, gh
                );
                continue;
            }
            let pred: Vec<u16> = sem.data.iter().copied().collect();
            accumulate_semantic(&mut acc, &pred, &gt);
            total += 1;
            handled = true;
        }
        if !handled {
            eprintln!("跳过（无语义输出）: {}", item.file);
            skipped += 1;
        }
    }

    if total == 0 {
        return Err(AppError::Config(format!(
            "没有可用的语义评估样本，请检查 dataset: {}",
            config.dataset.images
        )));
    }

    let report = json!({
        "model": config.model.path,
        "task": "semantic",
        "dataset": {
            "images": config.dataset.images,
            "annotations": config.dataset.annotations,
        },
        "num_images": total,
        "num_skipped": skipped,
        "eval": semantic_report(&acc, &names),
    });

    write_report(&config.eval.dir, &report)
}

/// 加载语义标签 JSON：`{"images": [{"file", "label"}]}`。
fn load_semantic_labels(dataset: &DatasetConfig) -> Result<SemanticLabels> {
    let text = std::fs::read_to_string(&dataset.annotations).map_err(|source| {
        AppError::ConfigIo {
            path: dataset.annotations.clone(),
            source,
        }
    })?;
    Ok(serde_json::from_str(&text)?)
}

/// 读取 8 位灰度标签 PNG，返回 `(宽, 高, 逐像素类别 id)`（行主序）。
fn load_semantic_label(path: &std::path::Path) -> Result<(u32, u32, Vec<u16>)> {
    let img = image::open(path)
        .map_err(|e| AppError::Config(format!("读取标签图失败 {}: {e}", path.display())))?;
    let gray = img.to_luma8();
    let (w, h) = (gray.width(), gray.height());
    let gt = gray.into_raw().into_iter().map(u16::from).collect();
    Ok((w, h, gt))
}

/// 逐像素累加每类的交 / 并（跳过 IGNORE）。
fn accumulate_semantic(acc: &mut HashMap<u16, (u64, u64)>, pred: &[u16], gt: &[u16]) {
    for (&p, &g) in pred.iter().zip(gt.iter()) {
        if p == SemanticMask::IGNORE || g == SemanticMask::IGNORE {
            continue;
        }
        if p == g {
            let e = acc.entry(p).or_insert((0, 0));
            e.0 += 1;
            e.1 += 1;
        } else {
            acc.entry(p).or_insert((0, 0)).1 += 1;
            acc.entry(g).or_insert((0, 0)).1 += 1;
        }
    }
}

/// 由交/并统计汇总 mIoU + per-class 明细。
fn semantic_report(acc: &HashMap<u16, (u64, u64)>, names: &HashMap<usize, String>) -> Value {
    let mut ids: Vec<u16> = acc.keys().copied().collect();
    ids.sort_unstable();

    let mut per_class = Vec::new();
    let mut sum = 0.0f32;
    for id in &ids {
        let &(inter, union) = &acc[id];
        let iou = if union == 0 {
            0.0
        } else {
            inter as f32 / union as f32
        };
        sum += iou;
        let name = names
            .get(&usize::from(*id))
            .cloned()
            .unwrap_or_else(|| id.to_string());
        per_class.push(json!({
            "class": id,
            "name": name,
            "inter": inter,
            "union": union,
            "iou": iou,
        }));
    }

    let n = ids.len();
    json!({
        "num_classes": n,
        "mIoU": if n > 0 { sum / n as f32 } else { 0.0 },
        "per_class": per_class,
    })
}

/// semantic 标签 JSON：`{"images": [{"file", "label"}]}`。
#[derive(Debug, Deserialize)]
struct SemanticLabels {
    #[serde(default)]
    images: Vec<SemanticItem>,
}

#[derive(Debug, Deserialize)]
struct SemanticItem {
    file: String,
    label: String,
}

/// 一个二维点（像素坐标）。
type Point = [f32; 2];

/// obb：DOTA 标注（8 个角点坐标）+ 旋转框 IoU 下的 mAP。
fn run_obb(model: &mut YOLOModel, config: &EvalConfig) -> Result<()> {
    let (images, gts) = load_obb_gts(&config.dataset)?;
    println!("评估集: {} 张图, {} 个 GT 实例", images.len(), gts.len());

    let images_dir = PathBuf::from(&config.dataset.images);
    let mut dets = Vec::new();
    let mut skipped = 0usize;
    for img in &images {
        let path = images_dir.join(&img.file_name);
        if !path.is_file() {
            eprintln!("跳过（图片不存在）: {}", path.display());
            skipped += 1;
            continue;
        }
        let results = model.predict(&path)?;
        dets.extend(collect_obb_detections(&results, img.id));
    }
    if skipped == images.len() {
        return Err(AppError::Config(format!(
            "评估集图片全部缺失，请检查 dataset.images 目录: {}",
            config.dataset.images
        )));
    }
    println!(
        "推理完成: {} 个预测实例（跳过 {} 张缺失图）",
        dets.len(),
        skipped
    );

    let report = json!({
        "model": config.model.path,
        "task": "obb",
        "dataset": {
            "images": config.dataset.images,
            "annotations": config.dataset.annotations,
        },
        "num_images": images.len() - skipped,
        "num_gt": gts.len(),
        "num_predictions": dets.len(),
        "eval": evaluate_obb(&gts, &dets),
    });

    write_report(&config.eval.dir, &report)
}

/// 从一次推理的 `Results` 中收集旋转框实例（4 个像素角点）。
fn collect_obb_detections(results: &[Results], image_id: i64) -> Vec<ObbDetection> {
    let mut dets = Vec::new();
    for r in results {
        let Some(obb) = &r.obb else { continue };
        let corners = obb.xyxyxyxy(); // (N,4,2) 像素坐标
        let conf = obb.conf();
        let cls = obb.cls();
        for i in 0..obb.len() {
            let class_id = cls[i] as usize;
            let class = r
                .names
                .get(&class_id)
                .cloned()
                .unwrap_or_else(|| class_id.to_string());
            let pts = (0..4)
                .map(|j| [corners[[i, j, 0]], corners[[i, j, 1]]])
                .collect();
            dets.push(ObbDetection {
                image_id,
                class,
                conf: conf[i],
                corners: pts,
            });
        }
    }
    dets
}

/// 加载 DOTA 数据集：`poly` 为 4 个角点的 `[x1,y1,x2,y2,x3,y3,x4,y4]` 像素坐标。
fn load_obb_gts(dataset: &DatasetConfig) -> Result<(Vec<CocoImage>, Vec<ObbGt>)> {
    let text = std::fs::read_to_string(&dataset.annotations).map_err(|source| {
        AppError::ConfigIo {
            path: dataset.annotations.clone(),
            source,
        }
    })?;
    let dota: DotaJson = serde_json::from_str(&text)?;
    let cat_name: HashMap<i64, String> = dota
        .categories
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();

    let gts = dota
        .annotations
        .iter()
        .filter_map(|ann| {
            let name = cat_name.get(&ann.category_id)?;
            let [x1, y1, x2, y2, x3, y3, x4, y4] = ann.poly;
            Some(ObbGt {
                image_id: ann.image_id,
                class: name.clone(),
                corners: vec![
                    [x1 as f32, y1 as f32],
                    [x2 as f32, y2 as f32],
                    [x3 as f32, y3 as f32],
                    [x4 as f32, y4 as f32],
                ],
            })
        })
        .collect();

    Ok((dota.images, gts))
}

/// obb 汇总：按类别计算旋转 IoU 下的 mAP。
fn evaluate_obb(gts: &[ObbGt], dets: &[ObbDetection]) -> Value {
    let mut dets_by_class: HashMap<String, Vec<ObbDetection>> = HashMap::new();
    for d in dets {
        dets_by_class.entry(d.class.clone()).or_default().push(d.clone());
    }
    let mut gts_by_class: HashMap<String, Vec<ObbGt>> = HashMap::new();
    for g in gts {
        gts_by_class.entry(g.class.clone()).or_default().push(g.clone());
    }

    let mut classes: Vec<String> = gts_by_class.keys().cloned().collect();
    classes.sort();

    let mut per_class = Vec::new();
    let mut sum_ap50 = 0.0f32;
    let mut sum_ap50_95 = 0.0f32;

    for cls in &classes {
        let cd = dets_by_class.get(cls).map(Vec::as_slice).unwrap_or(&[]);
        let cg = &gts_by_class[cls];
        let r = evaluate_obb_class(cd, cg);
        per_class.push(json!({
            "class": cls,
            "n_gt": r.n_gt,
            "ap50": r.ap50,
            "ap50_95": r.ap50_95,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
        }));
        sum_ap50 += r.ap50;
        sum_ap50_95 += r.ap50_95;
    }

    let n = classes.len();
    json!({
        "num_classes": n,
        "mAP50": if n > 0 { sum_ap50 / n as f32 } else { 0.0 },
        "mAP50_95": if n > 0 { sum_ap50_95 / n as f32 } else { 0.0 },
        "per_class": per_class,
    })
}

/// 单个类别的旋转 IoU AP 汇总（阈值网格同框 IoU：0.5..0.95）。
fn evaluate_obb_class(dets: &[ObbDetection], gts: &[ObbGt]) -> ClassReport {
    let mut aps = [0.0f32; IOU_THRESHOLDS.len()];
    let mut prf = (0.0f32, 0.0f32, 0.0f32);
    for (i, &t) in IOU_THRESHOLDS.iter().enumerate() {
        let (sorted, n_gt) = match_obb(dets, gts, t);
        let (ap, p, r, f1) = compute_ap(&sorted, n_gt);
        aps[i] = ap;
        if t == 0.5 {
            prf = (p, r, f1);
        }
    }
    ClassReport {
        n_gt: gts.len(),
        ap50: aps[0],
        ap50_95: aps.iter().sum::<f32>() / aps.len() as f32,
        precision: prf.0,
        recall: prf.1,
        f1: prf.2,
    }
}

/// 在某个旋转 IoU 阈值下做贪心匹配（图像内、同类）。
fn match_obb(
    dets: &[ObbDetection],
    gts: &[ObbGt],
    threshold: f32,
) -> (Vec<(f32, bool)>, usize) {
    let mut dets_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, d) in dets.iter().enumerate() {
        dets_by_image.entry(d.image_id).or_default().push(i);
    }
    let mut gts_by_image: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, g) in gts.iter().enumerate() {
        gts_by_image.entry(g.image_id).or_default().push(i);
    }

    let mut matched = vec![false; gts.len()];
    let mut out: Vec<(f32, bool)> = Vec::with_capacity(dets.len());

    for (image_id, det_indices) in &dets_by_image {
        let mut sorted = det_indices.clone();
        sorted.sort_by(|&a, &b| {
            dets[b]
                .conf
                .partial_cmp(&dets[a].conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let gt_indices = gts_by_image.get(image_id);
        for &di in &sorted {
            let d = &dets[di];
            let mut best: Option<(usize, f32)> = None;
            if let Some(gt_indices) = gt_indices {
                for &gi in gt_indices {
                    if matched[gi] {
                        continue;
                    }
                    let iou = rotated_iou(&d.corners, &gts[gi].corners);
                    if iou > best.map_or(0.0, |(_, b)| b) {
                        best = Some((gi, iou));
                    }
                }
            }
            match best {
                Some((gi, iou)) if iou >= threshold => {
                    matched[gi] = true;
                    out.push((d.conf, true));
                }
                _ => out.push((d.conf, false)),
            }
        }
    }

    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    (out, gts.len())
}

/// 鞋带公式：多边形的面积（顶点按环序）。
fn polygon_area(poly: &[Point]) -> f32 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    (s / 2.0).abs()
}

/// 若顶点为顺时针（有向面积为负）则反转为逆时针，便于裁剪时用「左半边」判内。
fn ensure_ccw(poly: &[Point]) -> Vec<Point> {
    let n = poly.len();
    let mut s = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    if s < 0.0 {
        poly.iter().rev().copied().collect()
    } else {
        poly.to_vec()
    }
}

/// 点 `(px, py)` 是否在由 `(ax, ay)→(bx, by)` 定向边的左侧（含边界）。
fn left_of(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> bool {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0.0
}

/// 线段 `s→e` 与 `a→b` 所在直线的交点（Sutherland-Hodgman 用）。
fn line_intersection(s: Point, e: Point, a: Point, b: Point) -> Point {
    let (x1, y1) = (s[0], s[1]);
    let (x2, y2) = (e[0], e[1]);
    let (x3, y3) = (a[0], a[1]);
    let (x4, y4) = (b[0], b[1]);
    let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if denom.abs() < 1e-9 {
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0];
    }
    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
}

/// 两个凸多边形（顶点按环序）的交集面积（Sutherland-Hodgman 裁剪 + 鞋带面积）。
fn intersect_area(a: &[Point], b: &[Point]) -> f32 {
    let clip = ensure_ccw(b);
    let mut output = a.to_vec();
    let m = clip.len();
    for i in 0..m {
        let j = (i + 1) % m;
        let (ax, ay) = (clip[i][0], clip[i][1]);
        let (bx, by) = (clip[j][0], clip[j][1]);
        if output.is_empty() {
            break;
        }
        let input = std::mem::take(&mut output);
        let n = input.len();
        for k in 0..n {
            let l = (k + 1) % n;
            let s = input[k];
            let e = input[l];
            let s_in = left_of(s[0], s[1], ax, ay, bx, by);
            let e_in = left_of(e[0], e[1], ax, ay, bx, by);
            if s_in {
                output.push(s);
            }
            if s_in != e_in {
                output.push(line_intersection(s, e, [ax, ay], [bx, by]));
            }
        }
    }
    polygon_area(&output)
}

/// 两个旋转框（凸四边形）的 IoU。
fn rotated_iou(a: &[Point], b: &[Point]) -> f32 {
    let area_a = polygon_area(a);
    let area_b = polygon_area(b);
    let inter = intersect_area(a, b);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// 一个旋转框预测实例（4 个像素角点）。
#[derive(Debug, Clone)]
struct ObbDetection {
    image_id: i64,
    class: String,
    conf: f32,
    corners: Vec<Point>,
}

/// 一个旋转框真值实例（4 个像素角点）。
#[derive(Debug, Clone)]
struct ObbGt {
    image_id: i64,
    class: String,
    corners: Vec<Point>,
}

/// DOTA 标注 JSON 的子集（与 COCO 同构，但 `annotations[].poly` 是 8 个角点坐标）。
#[derive(Debug, Deserialize)]
struct DotaJson {
    #[serde(default)]
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<DotaAnn>,
    #[serde(default)]
    categories: Vec<CocoCategory>,
}

#[derive(Debug, Deserialize)]
struct DotaAnn {
    image_id: i64,
    category_id: i64,
    poly: [f64; 8],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(image_id: i64, class: &str, conf: f32, bbox: [f32; 4]) -> Detection {
        Detection {
            image_id,
            class: class.to_string(),
            conf,
            bbox,
        }
    }

    fn gt(image_id: i64, class: &str, bbox: [f32; 4]) -> Gt {
        Gt {
            image_id,
            class: class.to_string(),
            bbox,
        }
    }

    #[test]
    fn ap_is_one_when_predictions_match_gt() {
        let dets = vec![
            det(0, "cat", 0.9, [10.0, 10.0, 20.0, 20.0]),
            det(0, "cat", 0.8, [30.0, 30.0, 40.0, 40.0]),
        ];
        let gts = vec![
            gt(0, "cat", [10.0, 10.0, 20.0, 20.0]),
            gt(0, "cat", [30.0, 30.0, 40.0, 40.0]),
        ];
        let r = evaluate_class(&dets, &gts);
        // 完美匹配时 recall 达到 1.0；ultralytics `compute_ap` 的哨兵会把
        // 末端 x=1.0 处插值到 0，故 AP 收敛到 0.995 而非 1.0（与 model.val() 对齐）。
        assert!((r.ap50 - 0.995).abs() < 1e-3, "ap50={}", r.ap50);
        assert!((r.ap50_95 - 0.995).abs() < 1e-3, "ap50_95={}", r.ap50_95);
        assert!((r.precision - 1.0).abs() < 1e-6);
        assert!((r.recall - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ap_is_zero_when_no_overlap() {
        let dets = vec![det(0, "cat", 0.9, [100.0, 100.0, 110.0, 110.0])];
        let gts = vec![gt(0, "cat", [0.0, 0.0, 10.0, 10.0])];
        let r = evaluate_class(&dets, &gts);
        assert_eq!(r.ap50, 0.0);
        assert_eq!(r.recall, 0.0);
    }

    #[test]
    fn ap_linear_interp_matches_ultralytics() {
        // 对照 ultralytics `compute_ap`（method="interp"）的 numpy 参考值。
        let cases = [
            (vec![0.0f32, 1.0], vec![0.0f32, 0.5], 0.4975f32),
            (vec![0.5f32, 1.0], vec![1.0f32, 1.0], 0.995f32),
            (vec![0.25, 0.5, 0.75, 1.0], vec![1.0, 0.75, 0.5, 0.25], 0.7175),
        ];
        for (recall, precision, expected) in cases {
            let ap = ap_linear_interp(&recall, &precision);
            assert!(
                (ap - expected).abs() < 1e-3,
                "recall={recall:?} precision={precision:?} ap={ap} expected={expected}"
            );
        }
    }

    #[test]
    fn matching_is_per_image() {
        // 跨图同名框不应匹配：图 0 的预测不能吃图 1 的真值。
        let dets = vec![det(0, "cat", 0.9, [10.0, 10.0, 20.0, 20.0])];
        let gts = vec![gt(1, "cat", [10.0, 10.0, 20.0, 20.0])];
        let r = evaluate_class(&dets, &gts);
        assert_eq!(r.recall, 0.0);
    }

    #[test]
    fn classify_topk_hits() {
        let top5 = vec![
            "minibus".to_string(),
            "police_van".to_string(),
            "trolleybus".to_string(),
        ];
        // top1 命中 → top1 与 top5 都命中。
        assert_eq!(topk_hits(&top5, "minibus"), (true, true));
        // 只在 top5 里 → top1 不命中，top5 命中。
        assert_eq!(topk_hits(&top5, "police_van"), (false, true));
        // 都不在 → 都不命中。
        assert_eq!(topk_hits(&top5, "cat"), (false, false));
    }

    #[test]
    fn oks_is_one_when_keypoints_identical() {
        let dt = vec![10.0f32, 20.0];
        let gt = vec![10.0f32, 20.0, 2.0];
        let s = oks(&dt, &gt, 100.0);
        assert!((s - 1.0).abs() < 1e-6, "oks={s}");
    }

    #[test]
    fn oks_drops_when_keypoint_far() {
        let dt = vec![30.0f32, 20.0];
        let gt = vec![10.0f32, 20.0, 2.0];
        let s = oks(&dt, &gt, 100.0);
        assert!(s < 0.1, "oks={s}");
    }

    #[test]
    fn pose_ap_and_pck_one_when_match() {
        let dets = vec![PoseDetection {
            image_id: 0,
            class: "person".to_string(),
            conf: 0.9,
            keypoints: vec![10.0, 20.0],
        }];
        let gts = vec![PoseGt {
            image_id: 0,
            class: "person".to_string(),
            area: 100.0,
            keypoints: vec![10.0, 20.0, 2.0],
        }];
        let r = evaluate_pose_class(&dets, &gts);
        // 完美匹配：ultralytics 哨兵使 AP 收敛到 0.995 而非 1.0。
        assert!((r.ap50 - 0.995).abs() < 1e-3, "ap50={}", r.ap50);
        assert!((r.ap50_95 - 0.995).abs() < 1e-3, "ap50_95={}", r.ap50_95);
        let pck = compute_pck(&dets, &gts);
        assert!((pck - 1.0).abs() < 1e-6, "pck={pck}");
    }

    #[test]
    fn mask_iou_full_and_disjoint() {
        let a = vec![1, 1, 0, 0];
        let b = vec![1, 1, 0, 0];
        assert!((mask_iou(&a, &b) - 1.0).abs() < 1e-6);
        let c = vec![0, 0, 1, 1];
        assert_eq!(mask_iou(&a, &c), 0.0);
    }

    #[test]
    fn rle_decode_roundtrip() {
        // counts [2,2,1,1] → [0,0,1,1,0,1]
        let mut mask = vec![0u8; 6];
        decode_rle(&mut mask, &[2, 2, 1, 1]);
        assert_eq!(mask, vec![0, 0, 1, 1, 0, 1]);
    }

    #[test]
    fn polygon_fill_rectangle() {
        let (h, w) = (6usize, 6usize);
        let mut mask = vec![0u8; h * w];
        fill_polygon(
            &mut mask,
            h,
            w,
            &[1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0],
        );
        // 内部像素中心在内，角点在外；4×4 共 16 像素被填。
        assert_eq!(mask[2 * w + 2], 1);
        assert_eq!(mask[0 * w + 0], 0);
        assert_eq!(mask.iter().filter(|&&v| v == 1).count(), 16);
    }

    #[test]
    fn semantic_accumulate_perfect() {
        // pred == gt → 每类 inter == union == 像素数。
        let mut acc = HashMap::new();
        let pred = vec![0u16, 1, 0, 1];
        let gt = vec![0u16, 1, 0, 1];
        accumulate_semantic(&mut acc, &pred, &gt);
        assert_eq!(acc[&0], (2, 2));
        assert_eq!(acc[&1], (2, 2));
    }

    #[test]
    fn semantic_accumulate_disjoint() {
        let mut acc = HashMap::new();
        let pred = vec![0u16, 1];
        let gt = vec![1u16, 0];
        accumulate_semantic(&mut acc, &pred, &gt);
        assert_eq!(acc[&0], (0, 2));
        assert_eq!(acc[&1], (0, 2));
    }

    #[test]
    fn rotated_iou_axis_aligned_overlap() {
        // A=(0,0)-(2,2)、B=(1,1)-(3,3)：交 1、并 7 → IoU = 1/7。
        let a = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let b = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let iou = rotated_iou(&a, &b);
        assert!((iou - 1.0 / 7.0).abs() < 1e-5, "iou={iou}");
    }

    #[test]
    fn rotated_iou_diamond_inside_square() {
        // 旋转 45° 的菱形（对角 2×2，面积 2）内接于 2×2 正方形（面积 4）→ IoU = 0.5。
        let square = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let diamond = vec![[1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]];
        let iou = rotated_iou(&square, &diamond);
        assert!((iou - 0.5).abs() < 1e-5, "iou={iou}");
    }

    #[test]
    fn rotated_iou_identical_and_disjoint() {
        let a = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        assert!((rotated_iou(&a, &a) - 1.0).abs() < 1e-6);
        let far = vec![[10.0, 10.0], [12.0, 10.0], [12.0, 12.0], [10.0, 12.0]];
        assert_eq!(rotated_iou(&a, &far), 0.0);
    }

    #[test]
    fn obb_ap_one_when_match() {
        let corners = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let dets = vec![ObbDetection {
            image_id: 0,
            class: "ship".to_string(),
            conf: 0.9,
            corners: corners.clone(),
        }];
        let gts = vec![ObbGt {
            image_id: 0,
            class: "ship".to_string(),
            corners,
        }];
        let r = evaluate_obb_class(&dets, &gts);
        // 完美匹配：ultralytics 哨兵使 AP 收敛到 0.995 而非 1.0。
        assert!((r.ap50 - 0.995).abs() < 1e-3, "ap50={}", r.ap50);
        assert!((r.ap50_95 - 0.995).abs() < 1e-3, "ap50_95={}", r.ap50_95);
    }
}
