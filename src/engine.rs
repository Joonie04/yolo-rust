//! 推理引擎：加载模型并驱动推理。

use std::time::Instant;

use ultralytics_inference::YOLOModel;

use crate::config::AppConfig;
use crate::error::{AppError, Result};
use crate::output::Output;
use crate::source::{open_source, resolve_source};

/// 推理引擎。
pub struct InferenceEngine;

impl InferenceEngine {
    /// 按配置执行完整推理流程。
    pub fn run(config: &AppConfig) -> Result<()> {
        // 1. 构建库的 InferenceConfig，并加载模型。
        let inference_config = config.inference.to_inference_config()?;
        let mut model = YOLOModel::load_with_config(&config.model.path, inference_config)?;
        let task = model.task().as_str().to_string();

        println!(
            "模型: {} | 任务: {} | 设备: {}",
            config.model.path,
            task,
            model.execution_provider()
        );

        // 2. 准备输入源与输出器（视频源写 mp4，图片源逐张存图）。
        let source = resolve_source(&config.source)?;
        let is_video = source.is_video();
        // 未编译 `video` feature 时，视频/流迭代器会无限返回错误，提前拦截并给出明确提示。
        if is_video && !cfg!(feature = "video") {
            return Err(AppError::Config(
                "视频/流输入需要 --features video（当前构建未启用该 feature）".into(),
            ));
        }
        let mut output = Output::new(&config.output, config.logging.verbose, &task, is_video)?;

        // 3. 遍历输入源：单张失败不中断，记录失败后继续。
        let stride = config.source.stride.unwrap_or(1).max(1);
        let max_frames = config.source.max_frames;
        let mut iter = open_source(source)?;
        let start = Instant::now();
        let mut processed = 0usize;
        let mut failed = 0usize;
        let mut total: Option<usize> = None;

        while let Some(item) = iter.next() {
            let (image, meta) = match item {
                Ok(ok) => ok,
                Err(e) => {
                    failed += 1;
                    output.record_failure("<解码失败>", &e.to_string());
                    continue;
                }
            };
            // 首个成功项携带总数（目录 / glob / 单图都来自库的 image_paths 长度）。
            if total.is_none() {
                total = meta.total_frames;
            }

            // 视频/流抽帧：按 stride 跳帧，达到 max_frames 后停止。
            if is_video {
                if meta.frame_idx % stride != 0 {
                    continue;
                }
                if let Some(max) = max_frames {
                    if meta.frame_idx >= max {
                        break;
                    }
                }
            }

            let path = meta.path.clone();
            match model.predict_image(&image, path.clone()) {
                Ok(results) => {
                    for result in results {
                        output.handle(&result, &image, &meta)?;
                        processed += 1;
                    }
                }
                Err(e) => {
                    failed += 1;
                    output.record_failure(&path, &e.to_string());
                }
            }

            Self::print_progress(start, processed, failed, total);
        }

        // 4. 落盘 JSON 并等待标注图 / 视频帧编码完成。
        output.finish()?;

        let done = processed + failed;
        println!("处理完成：成功 {} 张，失败 {} 张，共 {} 张。", processed, failed, done);
        if failed > 0 && config.logging.verbose {
            eprintln!("失败详情见 stderr 之外的 results.json（failures 字段）。");
        }
        Ok(())
    }

    /// 批量进度：`已完成/总数`、失败数、吞吐与 ETA（单图 / 未知总数时不打印）。
    fn print_progress(start: Instant, processed: usize, failed: usize, total: Option<usize>) {
        let show = match total {
            Some(t) => t > 1,
            None => true,
        };
        if !show {
            return;
        }
        let done = processed + failed;
        let rate = done as f64 / start.elapsed().as_secs_f64().max(1e-9);
        match total {
            Some(t) => {
                let eta = (t.saturating_sub(done)) as f64 / rate.max(1e-9);
                println!("进度 {done}/{t}（失败 {failed}）| {rate:.1} 图/s | ETA {eta:.0}s");
            }
            None => {
                println!("进度 {done} 张（失败 {failed}）| {rate:.1} 图/s");
            }
        }
    }
}
