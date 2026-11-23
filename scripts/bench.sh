#!/usr/bin/env bash
# 一键复现 CPU 工程基准：自动下载模型（若缺失）→ 跑 bench → 打印对比表。
#
# 需要 `yolo-rust` 在 PATH 上（预编译二进制，或 `cargo install --path .` 安装）。
# 开发环境可用 `cargo run -- bench config/bench_cpu.toml` 替代。
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="models/det/yolo26n.onnx"
if [ ! -f "$MODEL" ]; then
    echo "模型缺失，自动下载 yolo26n.onnx (~5MB) ..."
    mkdir -p models/det
    curl -fL --retry 3 -o "$MODEL" \
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.onnx"
fi

echo "运行 CPU 基准 ..."
yolo-rust bench config/bench_cpu.toml

echo ""
echo "=== 基准结果 ==="
python -I utils/bench_compare.py results/bench_cpu
