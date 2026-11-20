#!/usr/bin/env bash
# 服务器部署组装脚本：从源码目录「构建 + 组装」出可直接运行的部署目录。
#
# 用法（可在任意目录调用，脚本会自动 cd 到仓库根目录）：
#   ./deploy/assemble.sh                    # 默认：CUDA 构建 + 组装到 ~/autodl-tmp/yolo-rust-deploy
#   ./deploy/assemble.sh --tensorrt         # CUDA + TensorRT 构建 + 组装
#   ./deploy/assemble.sh --skip-build       # 不重编，只重新组装（复用已产出的二进制）
#   DEP=/path/to/deploy ./deploy/assemble.sh --tensorrt   # 指定部署目标目录
#
# 前置：
#   - 官方 ORT gpu_cuda13 已解压（默认 $HOME/autodl-tmp/project/onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib）
#   - --tensorrt 模式需已 `pip install tensorrt-cu13`（run.sh 会自动探测并加进 LD_LIBRARY_PATH，无需手设）
#
# 输出：自包含部署目录，含 yolo-rust + onnxruntime/lib + models(仅 onnx) + config + data + run.sh。
set -euo pipefail

# ---------- 可配置项（可用环境变量覆盖） ----------
DEP="${DEP:-$HOME/autodl-tmp/yolo-rust-deploy}"                                             # 部署目标目录
ORT_LIB="${ORT_LIB:-$HOME/autodl-tmp/project/onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib}"  # 官方 ORT 的 lib/
FEATURES="cuda"                                                                             # 构建 feature（--tensorrt 覆盖）
SKIP_BUILD=0
DATA_EXCLUDE="coco2017 eval equiv"                                                          # data/ 中不拷贝的评估大目录

# ---------- 参数解析 ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --tensorrt)   FEATURES="cuda tensorrt" ;;
    --cuda)       FEATURES="cuda" ;;
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0 ;;
    *) echo "错误: 未知参数 $1（见脚本头注释或 --help）" >&2; exit 2 ;;
  esac
  shift
done

# 从任意目录调用都能正确定位仓库根。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------- ① 构建（可跳过） ----------
if [ "$SKIP_BUILD" -eq 0 ]; then
  export ORT_LIB_LOCATION="$ORT_LIB"
  export ORT_PREFER_DYNAMIC_LINK=1
  echo ">> 构建: cargo build --release --features \"$FEATURES\""
  cargo build --release --features "$FEATURES"
else
  echo ">> 跳过构建（--skip-build），复用现有二进制"
fi

# ---------- ② 组装 ----------
# 官方 ORT lib 目录必须含 libonnxruntime.so，否则布局会错。
if [ ! -f "${ORT_LIB%/}/libonnxruntime.so" ]; then
  echo "错误: $ORT_LIB 下找不到 libonnxruntime.so —— ORT_LIB 应指向 gpu_cuda13 解压出的 lib/ 目录。" >&2
  exit 1
fi

echo ">> 组装到 $DEP"
mkdir -p "$DEP"/{onnxruntime,models,config,data,results}

# 二进制
cp target/release/yolo-rust "$DEP/"

# ORT 动态库（官方 gpu_cuda13 的 lib/，含 libonnxruntime.so 及 provider .so）
cp -r "${ORT_LIB%/}" "$DEP/onnxruntime/"

# 模型：整体拷，删除 .pt 只留 .onnx
cp -r models "$DEP/"
find "$DEP/models" -name '*.pt' -delete

# 配置：全部 toml
cp config/*.toml "$DEP/config/"

# 数据：拷样例，排除超大评估集（coco2017/eval/equiv）
for entry in data/*; do
  [ -e "$entry" ] || continue
  name="$(basename "$entry")"
  case " $DATA_EXCLUDE " in
    *" $name "*) echo "  跳过 data/$name（评估集，部署用不到）"; continue ;;
  esac
  cp -r "$entry" "$DEP/data/"
done

# 入口脚本 + Docker 打包文件（后者留给支持 Docker 的服务器/Jetson）
cp deploy/run.sh "$DEP/"
chmod +x "$DEP/run.sh"
cp deploy/Dockerfile deploy/.dockerignore "$DEP/" 2>/dev/null || true

# TensorRT 前置检查：确认 pip 的 tensorrt-cu13 已装（run.sh 运行时靠它找 libnvinfer.so.10）。
if [ "$FEATURES" = "cuda tensorrt" ] && command -v python3 >/dev/null 2>&1; then
  if ! python3 -c 'import tensorrt_libs' >/dev/null 2>&1; then
    echo "⚠️  警告: 未检测到 tensorrt-cu13（python3 -c 'import tensorrt_libs' 失败）；" >&2
    echo "    TensorRT EP 运行时将找不到 libnvinfer.so.10，请先 pip install tensorrt-cu13。" >&2
  fi
fi

# ---------- 完成 ----------
echo ""
echo "✅ 组装完成：$DEP"
echo "   二进制 feature：$FEATURES"
echo "   模型（仅 onnx）：$(find "$DEP/models" -name '*.onnx' | wc -l) 个"
echo ""
echo "   下一步："
echo "     cd $DEP"
echo "     ./run.sh bench config/bench_cuda.toml"
if [ "$FEATURES" = "cuda tensorrt" ]; then
  echo "     ./run.sh bench config/bench_tensorrt_fp16.toml"
fi
echo "     ./run.sh config/detect_cuda.toml"
