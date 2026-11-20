//! 二进制入口：解析子命令与配置路径，调用 `yolo_rust`。
//!
//! - `yolo-rust <config.toml>`              → 推理（批量目录 / glob 也走这里）
//! - `yolo-rust bench <config.toml>`        → 工程基准，输出 `bench_report.json`
//! - `yolo-rust eval <config.toml>`         → 精度评估，输出 `eval_report.json`

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let first = args.next();

    if first.as_deref() == Some("bench") {
        let config_path = args
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("config/bench.toml"));
        return report(yolo_rust::bench(&config_path));
    }

    if first.as_deref() == Some("eval") {
        let config_path = args
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("config/eval.toml"));
        return report(yolo_rust::eval(&config_path));
    }

    let config_path = first
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config/config.toml"));
    report(yolo_rust::run(&config_path))
}

fn report(result: yolo_rust::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("错误: {err}");
            ExitCode::FAILURE
        }
    }
}
