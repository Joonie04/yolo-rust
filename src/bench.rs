//! 工程基准：延迟百分位、吞吐、峰值内存，落盘 `bench_report.json`。
//!
//! 用法：`cargo run -- bench config/bench.toml`。先把整批输入载入内存（排除磁盘 I/O
//! 对计时的污染），预热若干遍后计时并采样进程峰值 RSS。`batch > 1` 时走
//! `predict_batch` 分块推理，`batch = 1` 时走 `predict_image` 逐张推理。

use std::path::PathBuf;
use std::time::Instant;

use image::DynamicImage;
use serde_json::{json, Value};
use sysinfo::ProcessesToUpdate;
use ultralytics_inference::{Speed, YOLOModel};

use crate::config::{BenchConfig, SourceConfig};
use crate::error::{AppError, Result};
use crate::source::{open_source, resolve_source};

/// 运行基准测试，并把报告写入 `<bench.dir>/bench_report.json`。
pub fn run_bench(config: &BenchConfig) -> Result<()> {
    let inference_config = config.inference.to_inference_config()?;
    let mut model = YOLOModel::load_with_config(&config.model.path, inference_config)?;
    let task = model.task().as_str().to_string();
    let device = model.execution_provider().to_string();

    println!(
        "基准模型: {} | 任务: {} | 设备: {}",
        config.model.path, task, device
    );

    // 1. 整批载入内存，避免磁盘 I/O 污染计时。
    let (images, paths) = load_images(&config.source)?;
    if images.is_empty() {
        return Err(AppError::Config("基准输入集为空（没有可推理的图像）".into()));
    }
    let batch_size = config.inference.batch.unwrap_or(1);
    println!(
        "输入 {} 张 | batch = {} | 预热 {} 遍 | 计时 {} 遍",
        images.len(),
        batch_size,
        config.bench.warmup,
        config.bench.iterations
    );

    // 2. 预热。
    for _ in 0..config.bench.warmup {
        run_pass(&mut model, &images, &paths, batch_size)?;
    }

    // 3. 计时 + 采样峰值 RSS。
    let pid = sysinfo::Pid::from_u32(std::process::id());
    let mut sys = sysinfo::System::new();
    let mut lat = Latency::new();
    let mut peak_rss: u64 = 0;
    let start = Instant::now();

    for _ in 0..config.bench.iterations {
        lat.push_all(&run_pass(&mut model, &images, &paths, batch_size)?);
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        if let Some(p) = sys.process(pid) {
            peak_rss = peak_rss.max(p.memory());
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let calls = lat.len();

    // 4. 汇总报告。
    let report = json!({
        "model": config.model.path,
        "task": task,
        "device": device,
        "batch_size": batch_size,
        "num_images": images.len(),
        "warmup": config.bench.warmup,
        "iterations": config.bench.iterations,
        "total_calls": calls,
        "latency_ms": {
            "preprocess": percentile_stats(&lat.preprocess),
            "inference": percentile_stats(&lat.inference),
            "postprocess": percentile_stats(&lat.postprocess),
            "total": percentile_stats(&lat.total),
        },
        "throughput": {
            "images_per_sec": calls as f64 / elapsed.max(1e-9),
            "total_sec": elapsed,
        },
        "memory": {
            "peak_rss_bytes": peak_rss,
            "peak_rss_mib": peak_rss as f64 / (1024.0 * 1024.0),
        },
    });

    let dir = PathBuf::from(&config.bench.dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("bench_report.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    println!("基准报告已保存: {}", path.display());

    let mut total_sorted = lat.total.clone();
    total_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "延迟 total p50/p95/p99 = {:.1}/{:.1}/{:.1} ms | 吞吐 {:.2} 图/s | 峰值 RSS {:.1} MiB",
        percentile(&total_sorted, 50.0),
        percentile(&total_sorted, 95.0),
        percentile(&total_sorted, 99.0),
        calls as f64 / elapsed.max(1e-9),
        peak_rss as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

/// 载入输入源的全部图像（含路径），供基准反复使用。
fn load_images(source: &SourceConfig) -> Result<(Vec<DynamicImage>, Vec<String>)> {
    let mut iter = open_source(resolve_source(source)?)?;
    let mut images = Vec::new();
    let mut paths = Vec::new();
    while let Some(item) = iter.next() {
        let (img, meta) = item?;
        images.push(img);
        paths.push(meta.path);
    }
    Ok((images, paths))
}

/// 跑一遍全部输入，返回每张图的耗时（不区分预热 / 计时）。
fn run_pass(
    model: &mut YOLOModel,
    images: &[DynamicImage],
    paths: &[String],
    batch_size: usize,
) -> Result<Vec<Speed>> {
    let mut speeds = Vec::new();
    if batch_size <= 1 {
        for (img, path) in images.iter().zip(paths) {
            let results = model.predict_image(img, path.clone())?;
            speeds.extend(results.iter().map(|r| r.speed.clone()));
        }
    } else {
        for start in (0..images.len()).step_by(batch_size) {
            let end = (start + batch_size).min(images.len());
            let results = model.predict_batch(&images[start..end], &paths[start..end])?;
            for batch in &results {
                speeds.extend(batch.iter().map(|r| r.speed.clone()));
            }
        }
    }
    Ok(speeds)
}

/// 各阶段耗时样本的累积器。
struct Latency {
    preprocess: Vec<f64>,
    inference: Vec<f64>,
    postprocess: Vec<f64>,
    total: Vec<f64>,
}

impl Latency {
    fn new() -> Self {
        Self {
            preprocess: Vec::new(),
            inference: Vec::new(),
            postprocess: Vec::new(),
            total: Vec::new(),
        }
    }

    fn push(&mut self, speed: &Speed) {
        self.preprocess.push(speed.preprocess.unwrap_or(0.0));
        self.inference.push(speed.inference.unwrap_or(0.0));
        self.postprocess.push(speed.postprocess.unwrap_or(0.0));
        self.total.push(speed.total());
    }

    fn push_all(&mut self, speeds: &[Speed]) {
        for speed in speeds {
            self.push(speed);
        }
    }

    fn len(&self) -> usize {
        self.total.len()
    }
}

/// 一组延迟样本的统计摘要：count / mean / p50 / p95 / p99。
fn percentile_stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mean = if n == 0 {
        0.0
    } else {
        sorted.iter().sum::<f64>() / n as f64
    };
    json!({
        "count": n,
        "mean": mean,
        "p50": percentile(&sorted, 50.0),
        "p95": percentile(&sorted, 95.0),
        "p99": percentile(&sorted, 99.0),
    })
}

/// 最近秩（nearest-rank）百分位，输入须已排序。
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let k = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[k.saturating_sub(1).min(sorted.len() - 1)]
}
