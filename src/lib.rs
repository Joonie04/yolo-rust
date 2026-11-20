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
