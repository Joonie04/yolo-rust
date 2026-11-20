//! 二进制入口：解析子命令与配置路径，调用 `yolo_rust`。
//!
//! - `yolo-rust <config.toml>`                          → 推理（批量目录 / glob 也走这里）
//! - `yolo-rust --model x.onnx --image y.jpg`           → 一键单图推理（出框 + JSON）
//! - `yolo-rust bench <config.toml>`                    → 工程基准，输出 `bench_report.json`
//! - `yolo-rust eval <config.toml>`                     → 精度评估，输出 `eval_report.json`

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_usage();
        return ExitCode::from(2);
    }

    match args[0].as_str() {
        "bench" => {
            let path = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| "config/bench.toml".to_string());
            return report(yolo_rust::bench(&path));
        }
        "eval" => {
            let path = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| "config/eval.toml".to_string());
            return report(yolo_rust::eval(&path));
        }
        "-h" | "--help" => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    if args[0].starts_with('-') {
        // 一键 CLI：`--model` / `--image` 等 flag 合成内存配置。
        match parse_run_args(&args) {
            Ok(opts) => report(yolo_rust::run_one(&opts)),
            Err(msg) => {
                eprintln!("错误: {msg}\n");
                print_usage();
                ExitCode::from(2)
            }
        }
    } else {
        // 配置路径模式（缺省 = 推理）。
        report(yolo_rust::run(&args[0]))
    }
}

/// 解析一键推理的 flag。
fn parse_run_args(args: &[String]) -> Result<yolo_rust::RunOptions, String> {
    let mut opts = yolo_rust::RunOptions::default();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--model" | "-m" => opts.model = next_value(args, &mut i, flag)?,
            "--image" | "-i" => opts.image = next_value(args, &mut i, flag)?,
            "--device" | "-d" => opts.device = next_value(args, &mut i, flag)?,
            "--conf" => {
                let v = next_value(args, &mut i, flag)?;
                opts.conf = Some(
                    v.parse()
                        .map_err(|_| format!("--conf 需要数字，得到 '{v}'"))?,
                );
            }
            "--iou" => {
                let v = next_value(args, &mut i, flag)?;
                opts.iou = Some(
                    v.parse()
                        .map_err(|_| format!("--iou 需要数字，得到 '{v}'"))?,
                );
            }
            "-h" | "--help" => return Err("帮助信息见下方".to_string()),
            other => return Err(format!("未知参数: {other}")),
        }
        i += 1;
    }

    if opts.model.is_empty() {
        return Err("缺少 --model <模型路径>".to_string());
    }
    if opts.image.is_empty() {
        return Err("缺少 --image <图片路径>".to_string());
    }
    Ok(opts)
}

/// 取 flag 后紧跟的值（`i` 已指向 flag，取完后 `i` 前移一位）。
fn next_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} 需要一个值"))
}

fn print_usage() {
    println!("yolo-rust —— 配置驱动的 YOLO 推理");
    println!();
    println!("用法:");
    println!(
        "  yolo-rust <config.toml>                         用配置文件推理（image / directory / glob / video）"
    );
    println!("  yolo-rust --model <x.onnx> --image <y.jpg>      一键单图推理（出框 + JSON）");
    println!("  yolo-rust bench <config.toml>                   工程基准 → bench_report.json");
    println!("  yolo-rust eval <config.toml>                    精度评估 → eval_report.json");
    println!();
    println!("一键推理参数:");
    println!("  --model, -m <路径>    ONNX 模型路径（或库可识别的模型名，自动下载）");
    println!("  --image, -i <路径>    输入图片路径");
    println!("  --device, -d <设备>   cpu（默认）/ cuda:0 / directml:0");
    println!("  --conf <0~1>          置信度阈值");
    println!("  --iou <0~1>           NMS IoU 阈值");
    println!("  -h, --help            显示本帮助");
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
