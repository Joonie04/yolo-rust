#!/usr/bin/env bash
# 服务器部署入口：固化 ORT 动态库路径，并从部署根目录执行 yolo-rust。
#
# 用法（在部署根目录内，即本脚本所在目录）：
#   ./run.sh config/detect_cuda.toml                 # 推理（单图/目录/glob/视频/流）
#   ./run.sh bench config/bench_cuda.toml            # 工程基准 → bench_report.json
#   ./run.sh eval config/eval_coco.toml              # 精度评估 → eval_report.json
#
# 为什么要有它：config 里的 model.path / source.path / output.dir 都是「相对工作目录」，
# 所以必须先 cd 到部署根目录；同时 libonnxruntime.so 不在系统默认路径，需要 LD_LIBRARY_PATH。
set -euo pipefail

# 以脚本所在目录作为工作根，保证相对路径（models/ data/ results/）可解析。
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# ONNX Runtime 动态库（官方 gpu_cuda13 包解压出的 lib/，含 libonnxruntime.so 及 provider .so）。
export LD_LIBRARY_PATH="$APP_DIR/onnxruntime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# TensorRT（可选）：若装了 pip 的 tensorrt-cu13，把其 lib 目录（libnvinfer.so.10 等）加进路径。
# 没装 TensorRT 或没有 python3 时静默跳过。
if command -v python3 >/dev/null 2>&1; then
  TRT_LIB="$(python3 -c 'import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))' 2>/dev/null || true)"
  if [ -n "${TRT_LIB:-}" ] && [ -d "$TRT_LIB" ]; then
    export LD_LIBRARY_PATH="$TRT_LIB:$LD_LIBRARY_PATH"
  fi
fi

# CUDA 13 运行时库（libcudart.so.13 / libcublas.so.13 等）若不在默认路径，
# 取消下一行注释并按实际路径补充（裸机通常已在 /usr/local/cuda*/lib64，或 AutoDL 已配好）。
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

exec "$APP_DIR/yolo-rust" "$@"
