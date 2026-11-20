//! 基于 `ultralytics-inference` 的配置驱动 YOLO 推理项目。

mod bench;
mod config;
mod engine;
mod error;
mod eval;
mod output;
mod source;

pub use config::{AppConfig, BenchConfig, EvalConfig};
pub use error::{AppError, Result};

use config::{InferenceSettings, LoggingConfig, ModelConfig, OutputConfig, SourceConfig};
use std::path::Path;

/// 入口：加载配置并执行推理。
pub fn run(config_path: impl AsRef<Path>) -> Result<()> {
    let config = AppConfig::load(config_path)?;
    engine::InferenceEngine::run(&config)
}

/// 入口：加载基准配置并跑工程基准，输出 `bench_report.json`。
pub fn bench(config_path: impl AsRef<Path>) -> Result<()> {
    let config = BenchConfig::load(config_path)?;
    bench::run_bench(&config)
}

/// 入口：加载评估配置并算 detect mAP，输出 `eval_report.json`。
pub fn eval(config_path: impl AsRef<Path>) -> Result<()> {
    let config = EvalConfig::load(config_path)?;
    eval::run_eval(&config)
}

/// 一键单图推理的参数（`yolo-rust --model <x.onnx> --image <y.jpg>`）。
///
/// 与配置文件里的 `[model]` / `[source]` / `[inference]` 等价，只是免去写 TOML：
/// 覆盖「陌生人 10 分钟跑通自己一张图」的最小场景。
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// ONNX 模型路径（也接受库可识别的模型名，自动下载）。
    pub model: String,
    /// 输入图片路径。
    pub image: String,
    /// 设备字符串：`cpu`（默认）/ `cuda:0` / `directml:0`。
    pub device: String,
    /// 置信度阈值（`None` = 库默认）。
    pub conf: Option<f32>,
    /// NMS IoU 阈值（`None` = 库默认）。
    pub iou: Option<f32>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            model: String::new(),
            image: String::new(),
            device: "cpu".to_string(),
            conf: None,
            iou: None,
        }
    }
}

/// 入口：一键单图推理（不写配置文件，等价于 `config.toml` 的 `image` 场景）。
pub fn run_one(opts: &RunOptions) -> Result<()> {
    let config = AppConfig {
        model: ModelConfig {
            path: opts.model.clone(),
        },
        source: SourceConfig {
            source_type: "image".to_string(),
            path: opts.image.clone(),
            stride: None,
            max_frames: None,
        },
        inference: InferenceSettings {
            device: Some(opts.device.clone()),
            conf: opts.conf,
            iou: opts.iou,
            ..Default::default()
        },
        output: OutputConfig::default(),
        logging: LoggingConfig::default(),
    };
    engine::InferenceEngine::run(&config)
}
