# 服务器部署指南（阶段 6 · 服务器 CLI 打包）

> 接口形式决策：**服务器走 CLI 批处理**；HTTP/gRPC/FFI 等服务化留到 Jetson（见 `docs/roadmap.md` §6）。
> 目标环境：AutoDL Ubuntu 22.04（glibc 2.35）+ RTX 3080（10GB）+ driver 595（CUDA 13.2）。
> 已验证：官方 ORT 1.28.0 `gpu_cuda13` + `ORT_LIB_LOCATION` 动态链接，在 22.04 上构建/推理通过
> （`CUDAExecutionProvider`，19.7 ms/图、45 图/s，见 `results/bench_cuda_server/bench_report.json`）。

---

## 1. 接口约定（job 约定）

一个「job」= 一份 config + 一批输入 → 一个带时间戳的输出目录。三种命令：

| 命令 | 作用 | 输出 |
|------|------|------|
| `yolo-rust <config.toml>` | 推理（单图/目录/glob/视频/流） | `results/<task>/<时间戳>/results.json` + 标注图/视频 |
| `yolo-rust bench <config.toml>` | 工程基准 | `<bench.dir>/bench_report.json` |
| `yolo-rust eval <config.toml>` | 精度评估 | `<eval.dir>/eval_report.json` |

- **输入**：`[source]` 的 `type` 决定 `image / directory / glob / video / webcam / stream`。
- **输出目录**：推理结果固定落在 `results/<task>/<unix秒时间戳>/`（`[output].dir` 只改根，`<task>/<时间戳>` 由代码追加，见 `src/output.rs:build_run_dir`）。
- **失败不中断**：批量/目录模式下单张失败记入 `results.json` 的 `failures` 字段，继续处理。

## 2. 相对路径约束

config 里的 `model.path`、`source.path`、`[output].dir` 都是**相对工作目录**的路径，所以二进制必须从「部署根目录」启动，且该目录下要有 `models/`、`data/`、`results/` 等。`deploy/run.sh` 已用 `cd "$(dirname "$0")"` 处理，直接 `./run.sh ...` 即可。

## 3. 环境变量（两类）

| 变量 | 时机 | 作用 |
|------|------|------|
| `ORT_LIB_LOCATION` | **构建期** | 指向官方 ORT 解压后的 `lib/`，让 ort-sys 跳过下载、动态链 `.so` |
| `ORT_PREFER_DYNAMIC_LINK=1` | **构建期** | 动态链接（官方包是 `.so` 不是 `.a`） |
| `LD_LIBRARY_PATH` | **运行期** | 指向部署目录的 `onnxruntime/lib/`，让二进制找到 `libonnxruntime.so` |
| CUDA 13 运行时库 | **运行期** | `libcudart.so.13`/`libcublas.so.13` 等，来自系统 CUDA 或 Docker 基镜像 |

> 构建期变量在 `cargo build` 前 export；运行期只需 `LD_LIBRARY_PATH`（run.sh 已固化）。

## 4. 部署目录布局

```
yolo-rust-deploy/                  # 部署根目录（= 容器内 /app）
├── yolo-rust                      # 目标二进制（target/release/yolo-rust）
├── onnxruntime/
│   └── lib/                       # 官方 gpu_cuda13 包解压的 lib/（.so 全家）
├── models/                        # 只拷要用的 *.onnx（不含 .pt）
├── config/                        # *.toml
├── data/                          # 输入样例 / bench 集（可直接跑，也可运行时挂载）
├── results/                       # 输出（运行时生成，可挂载持久化）
├── run.sh                         # 入口（固化 LD_LIBRARY_PATH + cd 到根目录）
├── Dockerfile                     # 打包成镜像用
└── .dockerignore
```

## 5. 直接部署（AutoDL 裸机，已验证可用）

> `$DEP` 是占位变量，只在**当前 shell** 有效；新开 SSH 会话要重新赋值（或用绝对路径）。
> 构建（①）只需做一次；之后每次只要「组装 + 跑」（②③）。
> **一键脚本**：`deploy/assemble.sh` 自动完成下面 ①②（构建 + 组装）——`--tensorrt` 构建 CUDA+TensorRT、`--skip-build` 只重组装，并自动排除 `data/` 里的 `coco2017/eval/equiv`。手动 ①② 仅作参考。

```bash
# ① 构建（一次性，ORT_LIB_LOCATION 指到官方 ORT 解压目录）
cd ~/autodl-tmp/project/yolo_rust
export ORT_LIB_LOCATION=$HOME/autodl-tmp/project/onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib
export ORT_PREFER_DYNAMIC_LINK=1
cargo build --release --features cuda

# ② 组装部署目录
DEP=~/autodl-tmp/yolo-rust-deploy
mkdir -p $DEP/{onnxruntime,models,config,data,results}
cp target/release/yolo-rust                                  $DEP/
cp -r $HOME/autodl-tmp/project/onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib $DEP/onnxruntime/
cp -r models $DEP/                                           # 若混入 .pt：find $DEP/models -name '*.pt' -delete
cp config/*.toml $DEP/config/
cp -r data/bench $DEP/data/                                  # 输入集（bench 用 24 张图）
cp data/bus.jpg $DEP/data/                                   # 单图样例（detect 用）
# 其它输入按需拷（如 video 样例 data/sample.mp4），或整个 data/（排除 coco2017 大目录）
cp deploy/run.sh $DEP/
chmod +x $DEP/run.sh

# ③ 跑（先 bench 验证 GPU，再跑任务）
cd $DEP
./run.sh bench config/bench_cuda.toml
./run.sh config/detect_cuda.toml
```

## 6. Docker 部署（可选，一键起）

> ⚠️ AutoDL 容器本身就跑在 Docker 里，**不支持嵌套 Docker**（`docker` 命令不存在）。
> 在 AutoDL 上请走 §5 裸机直跑；Docker 路径适用于自带 Docker 的 GPU 服务器或 Jetson。

前置：先按 §5 把部署目录组装好（二进制 + ORT 库 + models + config），再在部署目录内构建镜像。

```bash
cd $DEP
docker build -t yolo-rust:gpu .
docker run --gpus all \
  -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
  yolo-rust:gpu config/detect_cuda.toml
```

- 基镜像 `nvidia/cuda:13.2-runtime-ubuntu22.04` 自带 CUDA 13 运行时库（cudart/cublas/cublasLt），满足 ORT `gpu_cuda13` 的运行期依赖。
- 若该 tag 不存在，换 `13.1-runtime-ubuntu22.04` / `13.0-runtime-ubuntu22.04`（driver 13.2 向后兼容 ≤13.2 的运行时）。
- cuDNN 在 ORT 1.28.0 里是**运行期可选**；若想榨满 conv 性能，改基镜像为 `nvidia/cuda:13.2-cudnn-runtime-ubuntu22.04`。

## 7. 验证清单

- [ ] `./run.sh bench config/bench_cuda.toml` → `bench_report.json` 的 `device = CUDAExecutionProvider`，inference p50 ≈ 19 ms（而非 57 ms 的 CPU 兜底）。
- [ ] `./run.sh config/detect_cuda.toml` → `results/detect/<时间戳>/results.json` 非空、`detections` 数量 > 0。
- [ ] 批量目录含坏图不崩溃，`failures` 字段记录坏图。
- [ ] **全任务冒烟**：`./run.sh config/{segment,obb,pose,classify,depth,semantic}_cuda.toml` 各跑一遍 → 日志 `设备: CUDAExecutionProvider`，且 `results/<task>_cuda/<时间戳>/results.json` 的 `detections` / `masks` / `keypoints` / `obb` / `probs` / `semantic` / `depth` 对应字段非空（检测类任务数量 > 0）。

## 8. TensorRT EP（FP32 / FP16，已实测通过）

> FP16 是当前唯一可用的运行时量化（`int8` 在 vendored crate 里**只解析、无推理路径**，会回退 FP32）。
> 实测（AutoDL RTX 3080 10GB，yolo26n，24 图，2026-11-16 复测）：CUDA FP32 total p50 15.7ms/吞吐 55.9 图/s ≈ TensorRT FP32 15.6ms/56.8 图/s；TensorRT FP16 19.4ms/48.4 图/s。⚠️ 本卡 + nano 模型上 TensorRT 已无优势（TRT FP32 与 CUDA 持平、FP16 反而更慢），建议默认 CUDA FP32；TRT 收益需在更大模型 / 更大 `imgsz` 上重新评估。

**依赖**：① 官方 `gpu_cuda13` 包自带 `libonnxruntime_providers_tensorrt.so`（已确认）；② 系统 cuDNN 9（AutoDL 自带）；③ TensorRT 10 **CUDA 13 变体**（`libnvinfer.so.10` 等，需自行安装）。

```bash
# ① 装 TensorRT 10（CUDA 13 变体）。要求 Python 3.10~3.13。
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tensorrt-cu13

# ② 找到 libnvinfer.so.10 落点（pip wheel 放在 site-packages/tensorrt_libs/）
TRT_LIB=$(python3 -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))")
ls "$TRT_LIB" | grep -E 'libnvinfer.so.10|libnvonnxparser.so.10|libnvinfer_plugin.so.10'   # 三个都要有

# ③ 重编二进制（在源目录，加 tensorrt feature）
cd ~/autodl-tmp/project/yolo_rust
export ORT_LIB_LOCATION=$HOME/autodl-tmp/project/onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib
export ORT_PREFER_DYNAMIC_LINK=1
cargo build --release --features "cuda tensorrt"

# ④ 组装 + 跑（把新二进制和 config/bench_tensorrt*.toml、detect_tensorrt*.toml 拷进部署目录）
cp target/release/yolo-rust $DEP/
cp config/bench_tensorrt.toml config/bench_tensorrt_fp16.toml \
   config/detect_tensorrt.toml config/detect_tensorrt_fp16.toml $DEP/config/
# run.sh 已自动探测 tensorrt_libs 并加进 LD_LIBRARY_PATH，通常无需下面这行；仅当未自动识别时手工补：
# export LD_LIBRARY_PATH="$TRT_LIB:$LD_LIBRARY_PATH"
cd $DEP
./run.sh bench config/bench_tensorrt.toml              # 首次构建 FP32 引擎 ~1min
./run.sh bench config/bench_tensorrt_fp16.toml         # 首次构建 FP16 引擎
```

- **引擎缓存**：`models/det/.trt_cache/<stem>_{fp32,fp16}/`，首次跑构建（640×640 约 30s~1min），二次启动复用免重建。换模型/换 `imgsz`/换精度会各自建一份。
- **`LD_LIBRARY_PATH`**：TensorRT pip wheel 的 `.so` 不在系统默认路径，但 `deploy/run.sh` 已**自动探测**（`python3` 找 `tensorrt_libs` 落点并加进路径），无需手动 export。
- **版本兼容**：ORT 1.28.0 的 TensorRT provider 走 `libnvinfer.so.10` soname，任何 TensorRT 10.x 可链；但必须用 `-cu13` 变体（默认 `tensorrt` 包在新版也已默认 CUDA 13）。若引擎构建报版本错，回退 pin 旧版 `tensorrt-cu13==10.9.x`。
