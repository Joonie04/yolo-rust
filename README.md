# yolo-rust

基于 [`ultralytics-inference`](vendor/inference)（ONNX Runtime）的配置驱动 YOLO 推理项目。
通过 `config/*.toml` 调整要推理的模型、输入源、推理参数与输出形态，业务代码与推理逻辑解耦。

## 快速开始

**最快上手**：单图推理，无需写配置文件，直接出框 + JSON。

```bash
cargo run -- --model models/det/yolo26n.onnx --image data/bus.jpg
```

控制台打印 `4 persons, 1 bus` 这类检测摘要，并在 `results/detect/<时间戳>/` 落盘 `results.json` + 标注图。
（编译产物直接跑是 `yolo-rust --model <x.onnx> --image <y.jpg>`；可加 `--device cuda:0`、`--conf 0.25`、`--iou 0.7`。）

**预编译二进制（免装 Rust 工具链）**：到 [Releases](https://github.com/Joonie04/yolo-rust/releases) 下载对应平台的压缩包，解压后即可直接运行：

```bash
# 模型不存在时会自动下载（yolo26n.onnx ~5MB）；--image 换成你自己的图
yolo-rust --model yolo26n.onnx --image data/bus.jpg
```

两平台都是**单文件自包含**（静态链接 ONNX Runtime，解压后即可运行，无需外带任何 DLL）。GPU（CUDA/TensorRT/DirectML）需在目标机按 `docs/deploy.md` 从源码构建。

一键复现 CPU 基准：`scripts/bench.sh`（Linux/macOS）或 `scripts/bench.ps1`（Windows），自动下载模型 + 跑基准 + 出对比表。

**Docker**（免装任何东西）：`docker run --rm -v "$PWD:/data" ghcr.io/joonie04/yolo-rust --model yolo26n.onnx --image data/bus.jpg`；或本机 `docker build -t yolo-rust .` 自建镜像。

完整用法（配置文件，功能最全）：

```bash
# 1. 导出模型（.pt -> .onnx，需已装 ultralytics 的环境，如 yolo_algorithm_26）
#    python utils/export_onnx.py models/det/yolo26n.pt models/seg/yolo26n-seg.pt

# 2. 运行（CPU）
cargo run -- config/config.toml

# Windows GPU（DirectML）
cargo run --release --features directml -- config/config.toml

# NVIDIA GPU（CUDA / TensorRT）
cargo run --release --features cuda -- config/config.toml

# 3. 目录批量推理（进度 + 失败隔离）
cargo run -- config/detect_dir.toml

# 4. 工程基准（延迟百分位 / 吞吐 / 峰值 RSS → results/bench_report.json）
cargo run -- bench config/bench.toml

# 4a. GPU 基准（Windows DirectML，无需 CUDA/cuDNN）
cargo run --release --features directml -- bench config/bench_directml.toml

# 4b. GPU 基准（NVIDIA CUDA，需匹配的 CUDA 13 + cuDNN 9 运行时）
cargo run --release --features cuda -- bench config/bench_cuda.toml

# 4c. 汇总所有 bench 报告为 markdown 对比表
python -I utils/bench_compare.py

# 5. 精度评估（按模型任务自动分派 → results/eval_report.json）
cargo run -- eval config/eval.toml           # detect（COCO 标注 → mAP）
cargo run -- eval config/eval_classify.toml  # classify（标签 JSON → top1/top5）
cargo run -- eval config/eval_pose.toml      # pose（COCO 关键点 → OKS mAP + PCK@0.5）
cargo run -- eval config/eval_segment.toml   # segment（COCO 分割 → 掩码 IoU mAP）
cargo run -- eval config/eval_semantic.toml  # semantic（像素标签图 → mIoU）
cargo run -- eval config/eval_obb.toml       # obb（DOTA 旋转框 → 旋转 IoU mAP）

# 6. 视频推理（逐帧 → 标注 mp4 + 帧级 JSON；需 FFmpeg 开发库，见「视频」小节）
cargo run --release --features video -- config/video.toml

# 7. 统一报告（汇总 eval 精度 + bench 工程指标 + Python 参考 → docs/report.md）
python -I utils/gen_report.py
```

输出写入 `results/<task>/<时间戳>/`：每张图的标注图（`*.jpg`）与汇总 `results.json`；
批量模式下失败明细记录在 `results.json` 的 `failures` 字段，不中断整批。

## 配置

配置使用 TOML，字段说明见 `config/config.toml` 内注释与 `docs/plan.md`。

## 任务示例（7 类全任务）

每个任务一个配置，模型用 `models/<task>/` 下的 `.onnx`（本地 `.pt` 可用 `utils/export_onnx.py` 导出）：

| 任务 | 配置 | 模型 | 样例图 |
|------|------|------|--------|
| 目标检测 | `config/detect.toml` | `det/yolo26n.onnx` | `data/bus.jpg` |
| 实例分割 | `config/segment.toml` | `seg/yolo26n-seg.onnx` | `data/bus.jpg` |
| 语义分割 | `config/semantic.toml` | `sem/yolo26n-sem.onnx` | `data/bus.jpg` |
| 深度估计 | `config/depth.toml` | `depth/yolo26m-depth.onnx` | `data/depth/kids.jpg` |
| 图像分类 | `config/classify.toml` | `cls/yolo26n-cls.onnx` | `data/bus.jpg` |
| 姿态估计 | `config/pose.toml` | `pose/yolo26n-pose.onnx` | `data/zidane.jpg` |
| 旋转目标框 | `config/obb.toml` | `obb/yolo26n-obb.onnx` | `data/boats.jpg` |

JSON 输出采用「公共信封 + 任务载荷」结构，各任务字段（`detections`/`masks`/`keypoints`/`obb`/`probs`/`semantic`/`depth`）说明见 `docs/roadmap.md` 阶段 1。

## 目录结构

```
src/            Rust 源码（config / engine / source / output / bench / eval / error）
config/         配置文件
examples/       开发用 Rust 示例（make_segment_gt / make_semantic_gt / make_obb_gt 生成合成 GT）
models/         ONNX 模型（.pt 导出后放这里）
data/           输入图片 / 视频 / COCO 评估数据
results/        生成物（标注图 + JSON + bench_report.json + eval_report.json，gitignore）
docs/           设计文档
utils/          模型导出、合成 COCO 生成等脚本
vendor/     ultralytics-inference 库（path 依赖）
```

## GPU 与性能（Windows RTX 3060 实测）

- **DirectML**（`--features directml`）：Windows 上的可用 GPU 后端，无需 CUDA/cuDNN 版本匹配。
  yolo26m 吞吐 2.66 → 44.43 img/s（≈16.7×），inference p50 367.5 → 16.4 ms。
- **CUDA**（`--features cuda`）：后端可加载可推理，但下载的预编译 ONNX Runtime 针对 **CUDA 13 + cuDNN 9** 构建；
  本机为 CUDA 12.8 + cuDNN 8 时 conv 内核回退 CPU、无加速。目标机需装 CUDA 13 + cuDNN 9，或用 `ORT_LIB_PATH` 自编译 CUDA 12 版。
- **TensorRT / int8 / 服务器-Jetson 实测**：需 CUDA 13 + TensorRT 10 的目标环境，见 `docs/roadmap.md` 阶段 5。

## 视频

视频 / 摄像头 / RTSP 输入走 `type = "video" | "webcam" | "stream"`（`config/video.toml`），
逐帧推理后写回标注 mp4（或 `[output].save_frames = true` 逐帧存图）+ 帧级 JSON（含 `frame_idx`）。

FFmpeg 以**开发版（头文件 + 导入库 + 运行时 DLL）** vendor 进 `third_party/ffmpeg/`，项目自带、不依赖外部环境：

- `third_party/ffmpeg/include/` — 头文件（bindgen 生成绑定用）
- `third_party/ffmpeg/lib/` — 导入库（`.lib`，链接用）
- `third_party/ffmpeg/bin/` — 运行时 DLL（`build.rs` 在 `video` feature 下自动拷到 exe 旁，同 ort 的 `copy-dylibs`）

`.cargo/config.toml` 已把 `FFMPEG_DIR` 指向该目录，`cargo build --release --features video` 开箱即用。
写回 mp4 需要 H.264 编码器（libx264），因此 vendor 的是 gyan.dev 的 **GPL “full shared” 版**
`ffmpeg-9.0.2-full_build-shared.7z`（自带 libx264、DLL 自包含）；LGPL 版无 libx264，只能解码不能编码。

构建期额外需要 libclang（bindgen 用），位置由 `.cargo/config.toml` 的 `LIBCLANG_PATH` 指定；
它只参与编译，最终产物不依赖 libclang。

默认构建（不加 `--features video`）不受影响；视频源会在运行时报「需要 --features video」。

> 本机 Windows GPU（RTX 3060）上，视频推理建议用 `cargo run --release --features "video directml" -- config/video.toml`：
> 复用已缓存的 DirectML ONNX Runtime 包（`video.toml` 仍写 `device = "cpu"`，DirectML 包同样含 CPU EP）；
> 若只加 `--features video`，会另行联网下载 CPU 版 ONNX Runtime 包。

## 依赖与许可

- 底层库 `ultralytics-inference` 为 AGPL-3.0，MSRV 1.92（edition 2024）。
- 首次构建会联网下载 ONNX Runtime 二进制。
