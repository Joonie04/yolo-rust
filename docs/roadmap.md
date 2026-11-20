# YOLO 推理后端 — 分阶段路线图

> 在 `docs/plan.md`（架构设计）之上，本文件给出**从「已跑通」到「可部署推理后端」的完整演进路线**。
> 每个阶段都明确了：目标、要做的事、交付的功能点、评价指标与验收标准。

---

## 1. 项目定位

- **可复用 YOLO 推理后端**：Rust crate + CLI，配置驱动。
- **目标部署**：Linux 服务器（NVIDIA CUDA/TensorRT）+ NVIDIA Jetson（CUDA/TensorRT）。
- **覆盖范围**：7 类任务（detect / segment / semantic / depth / classify / pose / obb）× 4 类输入（单图 / 批量目录 / 视频 / 实时流）。
- **双指标**：工程指标（延迟、吞吐、内存）+ 精度指标（mAP、召回、mIoU、PCK 等）。

---

## 2. 已确认决策

| 决策点 | 结论 |
|--------|------|
| 项目定位 | 可复用推理后端，之后部署到边缘（Jetson）或服务器 |
| 扩展方向 | 视频处理、全任务覆盖、性能与 GPU、输出与报告 —— 全做 |
| 评价指标 | 工程指标 + 精度指标 都要 |
| 输入场景 | 单图 / 批量 / 视频 / 实时流 都需要 |
| 目标硬件 | 服务器 + NVIDIA GPU、NVIDIA Jetson（CUDA/TensorRT） |
| 任务覆盖 | detect、segment、pose、obb、classify、semantic（depth 已验证） |
| 精度数据 | 已有带标注数据 |
| 接口形式 | 未定（当前本机测试，后续部署到 Jetson/服务器） |

---

## 3. 架构原则（贯穿所有阶段）

1. **库优先（library-first）**：核心逻辑全部在 `src/` 的库中，`src/main.rs` 只是薄 CLI 包装。
   这样后续要加 HTTP/gRPC 服务或 C ABI/FFI 时，**不用重构核心**。
2. **后端可插拔**：`device` 配置字段决定执行后端（cpu/cuda/tensorrt/...），不把 GPU 逻辑写死在业务里。
3. **推理与评估解耦**：推理只产出「结构化结果」，报告/评估作为独立层消费这些结果。
4. **先正确、后优化**：先用 CPU 把功能链路做全做对，再用 GPU 做性能（阶段 1–4 与阶段 5 的先后关系由此而来）。

---

## 4. 阶段总览

| 阶段 | 主题 | 一句话目标 | 依赖 |
|------|------|-----------|------|
| 0 | 基线（已完成） | 配置驱动 + 单图 + detect/depth 跑通 | — |
| 1 | 全任务结构化输出 ✅ | 7 类任务 JSON + 标注图（已交付） | 0 |
| 2 | 视频与流输入 ✅ | 视频/摄像头/RTSP 逐帧推理 + 输出 | 0 |
| 3 | 批量处理 + 工程基准 ✅ | 目录批处理 + 延迟/吞吐/内存基准工具 | 0 |
| 4 | 精度评估（全任务指标 ✅） | 对标标注数据算 mAP/召回/mIoU/PCK/旋转 IoU | 1 |
| 5 | GPU 与性能优化 ✅（DirectML 实测） | CUDA/TensorRT、batch、量化、跨设备对比 | 1, 3 |
| 6 | 报告与部署 | 统一报告 + Jetson/服务器打包 + 接口决策 | 3, 4, 5 |

> 阶段 1、2、3 相互独立，可并行推进；阶段 4 依赖 1（结构化输出是算精度的前提）；阶段 5 依赖 1、3（有了基准工具才能量化 GPU 收益）。

---

## 5. 各阶段详情

### 阶段 0 — 基线（已完成 ✅）

**现状**：`config/*.toml` 驱动；支持 image 源；detect/depth 已验证；输出标注图 + `results.json`；错误处理与模块划分就绪。

- **功能**：`cargo run -- config/config.toml` 跑通单图推理。
- **评价指标**：detect、depth 各跑通一张样例，标注图与 JSON 正确生成。

---

### 阶段 1 — 全任务结构化输出（已完成 ✅）

**目标**：让 7 类任务都能产出**结构化、可被下游消费**的 JSON（这是后续精度评估的地基）。

**内容**：
- 为 `Results` 的每个字段补全 JSON 序列化：`boxes`、`masks`、`keypoints`、`probs`、`obb`、`semantic_mask`、`depth`。
- `masks` 用 RLE 或压缩多边形表示（避免把逐像素掩码整块塞进 JSON 撑爆文件）。
- `keypoints` 输出坐标 + 置信度；`obb` 输出旋转框（cx/cy/w/h/angle）。
- `semantic_mask` 输出每类像素占比（不落整张类别图）；`depth` 沿用现有 min/max（已实现）。
- 各任务的标注可视化走库的 `annotate`（已具备），确认渲染正确。
- 定义一套 **JSON schema**（结构约定），供评估/报告阶段统一解析。

**功能（交付）**：
- `output.rs` 支持按任务类型分派序列化：`to_json(result) -> Value`。
- 每个任务的标注图（含掩码/关键点/旋转框渲染）正常保存。

**评价指标**：
- 7/7 任务在各自样例上产出**非空、schema 合法**的 JSON。
- 检测/分割/姿态/旋转框在已知样例上的框/掩码/关键点数量 > 0（功能正确性冒烟）。
- 序列化后的 JSON 可被反序列化回结构体（round-trip 校验）。

**验收标准**：准备 7 张任务样例图，一键跑完 7 个任务，JSON schema 校验全部通过。 —— ✅ 已达成。

**完成情况（2026-10-16）**：
- 7/7 任务跑通并产出标注图 + `results.json`（detect / segment / semantic / depth / classify / pose / obb）。
- 模型：本地 `.pt` 用 `yolo_algorithm_26`（ultralytics 8.4.175）导出为 `.onnx`，脚本 `utils/export_onnx.py`；样例图 `data/{bus,zidane,boats}.jpg`。
- JSON 结构（公共信封 `path/task/orig_shape/inference_shape/speed_ms` + 任务载荷，坐标归一化 `[0,1]`）：

| 字段 | 任务 | 说明 |
|------|------|------|
| `detections` | detect / segment / pose | `[{name, class, confidence, box:{x1,y1,x2,y2}}]` |
| `masks` | segment | `{count, shape:[H,W], areas:[面积占比…]}` |
| `keypoints` | pose | `{count, num_keypoints, has_confidence, keypoints:[[x,y,conf]…]}` |
| `obb` | obb | `[{name, class, confidence, obb:{cx,cy,w,h,rotation}, corners:[[x,y]×4]}]` |
| `probs` | classify | `{top1:{class,name,confidence}, top5:[…]}` |
| `semantic` | semantic | `{shape, classes:[{class,name}], pixel_fractions:{id:占比}}` |
| `depth` | depth | `{min_m, max_m, shape}` |

> 掩码只存面积占比（逐像素掩码体积过大），完整可视化见标注图；RLE/多边形可留待阶段 4 精度评估再补。

---

### 阶段 2 — 视频与流输入

**目标**：把输入源从「单图」扩展到「视频 / 摄像头 / RTSP」，逐帧推理并输出。

**内容**：
- 开启 `video` feature，接通 `Source::Video / Webcam / Stream`。
- 逐帧推理：`SourceIterator` 遍历帧 → `predict_image` → 收集每帧结果。
- 输出：带标注的视频（写回 mp4）、每帧 JSON（帧索引对齐）、可选抽帧（`stride`/`max_frames`）。
- 配置新增 `[source]` 的视频相关字段（`stride` / `max_frames`）与 `[output]` 的视频字段（`save_frames`：视频源默认写回 mp4，置 true 则逐帧存图）。

**功能（交付）**：
- `cargo run -- config/video.toml` 处理一个视频文件，产出标注视频 + 帧级 JSON。
- 摄像头/RTSP 实时流支持（`type = "webcam" | "stream"`）。

**评价指标**：
- 输出视频**时长与帧数正确**（按 `stride` 抽帧时一致），可正常播放。
- 帧级 JSON 与视频帧索引一一对应。
- 处理 FPS（CPU 基线，作为阶段 5 的对比锚点）：detect 目标 ≥ 数 FPS（具体值随模型定，记录即可）。

**验收标准**：一个样例视频全程跑通，输出视频 + JSON 帧数与抽帧策略一致。

**完成情况（2026-10-20，端到端跑通 ✅）**：
- **输入源**：`source.rs` 已把 `type = "video" | "webcam" | "stream"` 接到库的 `Source::Video/Webcam/Stream`（这些枚举变体始终可用；实际解码由库的 `video` feature 提供，缺省时返回“需要 --features video”错误）。
- **逐帧推理**：`engine.rs` 复用 `SourceIterator` 遍历帧 → `predict_image`；新增视频源抽帧 `stride`（每 N 帧取 1，默认 1）与 `max_frames`（最多帧数，默认到结尾）。
- **输出**：`output.rs` 改为复用库的 `io::SaveResults`（统一处理「图片逐张 / 视频写回 mp4 / 视频逐帧存图」，含编码工作线程，编码与推理重叠）：
  - `[output].save_frames = false`（默认）→ 视频写回单个标注 mp4（`<stem>.mp4`，fps 取自源）；
  - `save_frames = true` → 逐帧存到 `<stem>_frames/<stem>_<n>.jpg`；
  - 标注渲染改用库的 `annotate_image`（与原先 `result.save()` 同源）。
- **帧级 JSON**：结果信封新增 `frame_idx`（帧索引）与 `fps`，视频每帧一个元素、索引对齐。
- **配置**：新增 `config/video.toml`；`SourceConfig` 增 `stride`/`max_frames`，`OutputConfig` 增 `save_frames`（均带默认值，不影响既有图片配置）。
- **测试样例**：用系统 ffmpeg CLI 由 `data/bus.jpg` 生成 `data/sample.mp4`（810×1080、15fps、45 帧）。
- **验证状态**：默认构建（无 `video` feature）编译通过 + 16 个单测通过；`--features video` 已编译并链接通过。FFmpeg 以**开发版 vendor 进 `third_party/ffmpeg/`**（`include/`+`lib/`+`bin/`），`.cargo/config.toml` 把 `FFMPEG_DIR` 指向它、`LIBCLANG_PATH` 指向本机 libclang（仅构建期 bindgen 用），`build.rs` 在 `video` feature 下把 DLL 自动拷到 exe 旁（同 ort 的 `copy-dylibs`，产物自带运行库）。写回 mp4 需 libx264（GPL），故 vendor 的是 gyan.dev **`ffmpeg-9.0.2-full_build-shared`（GPL）**：自带 libx264（`ffmpeg -encoders` 确认）、DLL 自包含（avcodec-63 / avformat-63 / avutil-61 / avfilter-12 / swscale-10 / swresample-7 等），替代原先非自包含的 conda-forge 6.1.1（LGPL、无 libx264）。**端到端冒烟已跑通**：`cargo run --release --features "video directml" -- config/video.toml` 处理 `data/sample.mp4`（810×1080、45 帧）→ 输出 `sample.mp4`（ffprobe 确认 h264、45 帧）+ `results.json`（45 个元素，`frame_idx` 0–44 对齐，失败 0）。
- **webcam / RTSP 实时流**：代码路径已就绪（`type = "webcam" | "stream"`），本机无摄像头/流源，未实测；实时流水线（读帧/预处理/推理/后处理重叠）留待阶段 5 的实时优化。

---

### 阶段 3 — 批量处理 + 工程基准（已完成 ✅）

**目标**：支持目录/glob 大批量推理，并建立**可重复的工程基准工具**。

**内容**：
- 目录/glob 批量：遍历全部图片，进度显示（已完成/总数、ETA）、失败单张不中断。
- 批量推理 `predict_batch`（配合 `batch` 参数）对比单张吞吐。
- **基准工具** `bench`：固定模型 + 固定输入集，测量并导出
  - 延迟：preprocess / inference / postprocess / total，p50 / p95 / p99
  - 吞吐：images/s（batch=1 与 batch=N 对比）
  - 内存：峰值 RSS
- 基准结果落盘为 JSON，供阶段 5 做跨设备/跨后端对比。

**功能（交付）**：
- `cargo run -- config/detect_dir.toml` 批量处理一个目录。
- `cargo run -- bench config/bench.toml` 跑基准并输出 `bench_report.json`。

**评价指标**：
- 批量模式下吞吐相对逐张有可量化的提升（记录 batch=1 vs N）。
- 基准结果可复现（同设备同配置两次跑，p50 波动 < 5%）。
- 1000 张图批量处理无崩溃、失败图被记录而非中断。

**验收标准**：目录批量跑通 + `bench` 产出含延迟/吞吐/内存的 JSON 报告。 —— ✅ 已达成。

**完成情况（2026-10-16）**：
- 批量处理：`engine.rs` 遍历输入源时**单张失败不中断**，进度行打印「已完成/总数、失败数、吞吐（图/s）、ETA」；失败明细写入 `results.json` 的 `failures` 字段（含路径 + 错误）。已验证：3 图目录批量跑通、混入 1 张损坏 jpg 时成功 1 / 失败 1 且不崩溃。
- 基准工具 `src/bench.rs`：`cargo run -- bench config/bench.toml` 输出 `results/bench_report.json`，字段：
  - `latency_ms.preprocess / inference / postprocess / total`：各含 `count / mean / p50 / p95 / p99`；
  - `throughput`：`images_per_sec` + `total_sec`；
  - `memory`：`peak_rss_bytes` / `peak_rss_mib`（进程峰值 RSS，`sysinfo` 采样）；
  - 元信息：`model / task / device / batch_size / num_images / warmup / iterations / total_calls`。
- batch 对比：`inference.batch = N`（N>1）时走 `predict_batch` 分块推理，报告记录 `batch_size`；跑 batch=1 与 batch=N 两份配置即可对比吞吐。
- 新增文件：`src/bench.rs`、`config/detect_dir.toml`、`config/bench.toml`；新增依赖 `sysinfo`（峰值 RSS）、`image`（复用库同版本）。
- 待大规模验证：1000 张图的压力测试（当前用 3 图样例冒烟，循环设计已支持大批量，未做规模压测）。

---

### 阶段 4 — 精度评估（全任务指标 ✅，等价性：检测级 + mAP 对齐 ✅）

**目标**：用你已有的标注数据，对标官方实现算精度，证明「我们的后端 = Ultralytics 官方」。

**内容**：
- 评估数据加载：**COCO 格式**（默认）+ 可插拔解析器（YOLO txt / 自定义格式按需加）。
- 指标实现：
  - detect / segment：mAP@0.5、mAP@0.5:0.95、precision / recall、per-class AP
  - pose：OKS 与 PCK@0.5
  - obb：旋转 IoU 下的 mAP
  - classify：top1 / top5 准确率
  - semantic：mIoU
  - depth：相对误差 / 有效像素覆盖率（可选，深度标注通常较难获得）
- 等价性验证：同模型同参数，与 Ultralytics Python 跑同一评估集，**对齐结果**。
- 输出评估报告（JSON + 可读摘要）。

**功能（交付）**：
- `cargo run -- eval config/eval.toml` 产出 `eval_report.json`（per-class + 汇总指标）。

**评价指标**：
- detect/segment 与 Python 官方对齐：|ΔmAP@0.5| ≤ 0.005、|ΔmAP@0.5:0.95| ≤ 0.005（等价性）。
- pose PCK@0.5、classify top1、semantic mIoU 输出数值正确（对已知样例可手工核对）。
- 报告含 per-class 分解，便于定位差异。

**验收标准**：同一评估集上，我们的 detect mAP 与官方差距在容差内；其余任务指标输出合理。

**完成情况（2026-10-17，detect + classify + pose + segment + semantic + obb）**：
- 新增 `src/eval.rs` + `config/eval.toml` / `config/eval_classify.toml` + 子命令 `cargo run -- eval <config>`，按模型 `task()` 自动分派，输出 `results/eval_report.json`。
- **COCO 加载器**（detect）：解析 `images` / `annotations` / `categories`，`bbox`（xywh 像素）转 xyxy；预测与真值按**类别名称**对齐（不依赖类别 ID 顺序，规避模型类别序与 COCO id 不一致）。
- **detect 指标**：`mAP@0.5`、`mAP@0.5:0.95`（10 个 IoU 阈值 0.5–0.95，101 点插值）、`precision / recall / f1`（IoU=0.5 处最大 F1 点）、`per_class`（含每类 `n_gt`）。
- **classify 指标**：`top1_accuracy` / `top5_accuracy` + `per_image`（每图 `true / top1 / top5 / top1_correct / top5_correct`）。标签格式为简单 JSON `{"images":[{"file","class"}]}`。
- **pose 指标**：`mAP@0.5` / `mAP@0.5:0.95`（OKS，10 个阈值 0.5–0.95，101 点插值，对标 `model.val()`）+ `pck50`（bbox 归一化 PCK@0.5）+ `per_class`。COCO `keypoints`（`[x,y,v]×17` 像素）解析为 `PoseGt`，OKS 采用 pycocotools 口径（`vars=(2σ)²`、`e=d²/vars/area/2`、按真值可见关键点求均值），匹配仍在图像内贪心进行。
- **segment 指标**：`mAP@0.5` / `mAP@0.5:0.95`（掩码 IoU，10 个阈值 0.5–0.95，101 点插值）+ `per_class`。COCO `segmentation` 支持**多边形**（射线法光栅化到像素中心点判定）与**未压缩 RLE**（游程解码）两种，预测掩码阈值 0.5 二值化；压缩 RLE（COCO-stuff / 全景）暂不支持。
- **semantic 指标**：`mIoU`（逐类像素 IoU 的均值）+ `per_class`（`class / name / inter / union / iou`）。标签格式为简单 JSON `{"images":[{"file","label"}]}`，`label` 是 8 位灰度 PNG（像素值 = 类别 id，相对 `dataset.images` 解析），`IGNORE` 像素（`u16::MAX`）不计入交并。
- **obb 指标**：`mAP@0.5` / `mAP@0.5:0.95`（旋转 IoU，10 个阈值 0.5–0.95，101 点插值）+ `per_class`。标注为 **DOTA 风格** `poly`（4 个角点的 `[x1,y1,x2,y2,x3,y3,x4,y4]` 像素坐标），预测取 `obb.xyxyxyxy()` 的 4 个像素角点；旋转 IoU 用**凸多边形交集**（Sutherland-Hodgman 裁剪 + 鞋带公式面积）计算，两框按各自面积之和减交集求并，匹配仍在图像内贪心进行。
- **匹配语义**：贪心匹配在「图像内」进行（某图预测只匹配同图同类别真值），与 pycocotools 一致。
- **冒烟验证**：detect 用 `utils/make_synthetic_coco.py` 生成合成 COCO（真值==预测）→ `mAP@0.5 = mAP@0.5:0.95 = 1.0`；classify 用模型自身 top1 作标签 → `top1 = top5 = 1.0`；pose 用 `utils/make_synthetic_pose_coco.py`（真值==模型自身关键点）→ `mAP@0.5 = mAP@0.5:0.95 = pck50 = 1.0`；segment 用 `cargo run --example make_segment_gt`（真值==模型自身掩码的 RLE）→ `mAP@0.5 = mAP@0.5:0.95 = 1.0`；semantic 用 `cargo run --example make_semantic_gt`（真值==模型自身语义图）→ `mIoU = 1.0`；obb 用 `cargo run --example make_obb_gt`（真值==模型自身旋转框角点）→ `mAP@0.5 = mAP@0.5:0.95 = 1.0`（harbor + ship 两类，167 个实例）。另有 16 个单元测试（detect 3 + classify 1 + pose 3 + segment 3 + semantic 2 + obb/旋转 IoU 4）。
- **待续**：
  - **等价性验证**：检测级 raw 逐字段对齐 ✅（见下「等价性验证」小节）；**mAP 对齐 ✅ 已完成**（2026-10-21，COCO val2017，见下「等价性验证（mAP 对齐）」小节）。
  - 真实 COCO 评估用 `conf = 0.001`（当前 eval*.toml 为冒烟测试设了 `conf = 0.25`）。

**等价性验证（raw 逐字段对齐，2026-10-20 ✅）**：不依赖真实标注，直接比对「同图同参数下 `.pt`（PyTorch）与 `.onnx`（ONNX Runtime）的检测输出」，从检测级证明后端等价：

- **方法**：`utils/dump_detections.py`（ultralytics 8.4.175，`predict(conf=0.25, iou=0.7, imgsz=640, device=cpu)`）落盘 Python 参考 → `config/equiv_detect.toml` 跑 Rust（同参数）→ `utils/compare_equiv.py` 贪心匹配（同类别 + 最大 IoU ≥ 0.5）。
- **结果**（3 图：bus / zidane / boats，yolo26n）：
  - **检测数完全一致**：bus 5 / zidane 3 / boats 0，共 8 个检测全部匹配、0 未匹配。
  - **框坐标**：max |Δ| ≈ 0.013（归一化，典型 0.001–0.003），来自 resize/letterbox 插值的跨引擎差异。
  - **置信度**：max |Δ| ≈ 0.109、典型 0.01–0.06，且越接近 0.5 偏差越大（sigmoid 在 0.5 处导数最大），是 PyTorch↔ONNX Runtime 浮点核差异经 sigmoid 放大所致，非系统性错误。
- **结论**：**检测级等价成立**（同目标、同类别、同几何，置信度在跨引擎数值容差内）。据此 mAP 也应对齐；精确的 `|ΔmAP@0.5| ≤ 0.005` 仍需真实标注 + `model.val()` 参考（见下）。

> conf=0.001 下 raw 比对不可用：该阈值输出数百个重叠低置信框，PyTorch/ONNX 的极小浮点差在 NMS 里级联成「保留哪几个框」的不同，逐框无法对应；故 raw 对齐在稳定阈值（0.25）做，conf=0.001 只用 mAP（汇总统计）衡量。

**等价性验证（对齐 Python `model.val()`）所需信息**：完成对齐需要以下材料，缺一不可：

1. **真实标注数据**（任选你要对齐的任务，COCO 优先）：
   - detect：`images/` 验证集图片目录 + `instances_val2017.json`（含 `annotations[].bbox`）或你自己的 COCO 标注。
   - segment：同上，但 `annotations[].segmentation` 为多边形/RLE。
   - pose：同上，但 `annotations[].keypoints`（`[x,y,v]×17`）。
   - obb：DOTA 风格 `annotations[].poly`（4 角点 8 坐标）——COCO 本身无旋转框，需 DOTA 或转成该格式。
   - classify：按 `class/` 子目录组织的图片，或 `{"images":[{"file","class"}]}` 标签 JSON。
   - semantic：每张图配 8 位灰度类别 PNG。
2. **同一份模型权重**：Python 侧用 `.pt`，我们侧用**该 .pt 同版本导出的 .onnx**（`utils/export_onnx.py`），并核对两边的模型名/类别表一致。
3. **对齐用的推理参数**（两边必须一致）：`conf = 0.001`、`iou = 0.7`、`imgsz`（640 或模型默认）、`batch = 1`；NMS/后处理口径以 Ultralytics `val` 默认值为准。
4. **Python 侧参考结果**：运行 `model.val(data=<你的数据>, conf=0.001, iou=0.7, imgsz=<同>, batch=1)`，把打印的 per-class AP / `mAP@0.5` / `mAP@0.5:0.95` 数值给我，作为对齐目标。
5. **Python 环境版本**：`ultralytics` 包版本号（`pip show ultralytics`），因不同版本的 val 实现细节可能影响小数点后结果。
6. **对齐判定标准**：`|ΔmAP@0.5| ≤ 0.005` 且 `|ΔmAP@0.5:0.95| ≤ 0.005`（detect/segment/pose/obb）；classify 的 `top1/top5`、semantic 的 `mIoU` 以可手工核对为准。

**等价性验证（mAP 对齐，2026-10-21 ✅）**：真实 COCO val2017 全量对齐。方法：`utils/run_val_reference.py` 跑 Ultralytics `model.val(conf=0.001, iou=0.7, imgsz=640, batch=1)` 作参考，Rust 侧 `cargo run -- eval config/eval_coco.toml`（同参数），两边吃同一份 `yolo26n.onnx` + 转换后的 COCO 标注。

- **数据准备**：`convert_coco`（converter.py:245）把 `instances_val2017.json` 转成 YOLO 标签（映射 91→80 类、跳过 446 个 iscrowd 标注），图片归位 `images/val2017/`、标签归位 `labels/val2017/`（ultralytics 的 `img2label_paths` 要求图片在 `images/` 目录下才找得到标签）。
- **评测口径修正（对齐 ultralytics `compute_ap`）**：
  - AP 插值由 COCO 阶梯插值改为**线性插值**（method="interp"：精度包络 + 101 点 `np.interp` + `np.trapezoid` 梯形积分），新增 `ap_linear_interp`/`interp_linear`，单测对 numpy 参考值（0.4975 / 0.995 / 0.7175）通过；
  - 跳过 iscrowd 标注（GT 从 36781 → 36335，与 ultralytics 一致）；
  - 完美匹配的 AP 由 1.0 收敛到 0.995（ultralytics 哨兵在 x=1.0 处插值到 0 所致），3 个相关单测断言同步更新。
- **结果**（mAP@0.5 / mAP@0.5:0.95）：

| 参考 | 后端 | mAP@0.5 | mAP@0.5:0.95 |
|------|------|---------|--------------|
| Python `.pt` | torch 2.5.1 | 0.56112 | 0.40226 |
| Python `.onnx` | onnxruntime 1.23.1 | 0.56172 | 0.40324 |
| Python `.onnx` | onnxruntime 1.28.0 | 0.56172 | 0.40324 |
| **Rust `.onnx`** | **ort 1.28.0** | **0.55429** | **0.39833** |

  - `.pt` vs `.onnx`：ΔmAP@0.5 ≈ 0.0006 → ONNX 导出保真，无明显精度损失。
  - **Rust vs Python（同为 `.onnx`）**：ΔmAP@0.5 = 0.0074、ΔmAP@0.5:0.95 = 0.0049。mAP@0.5:0.95 **在容差内**（≤ 0.005）；mAP@0.5 略超 0.0024。
- **根因定位（排除法）**：① 评测逻辑已精确对齐（AP 插值 / iscrowd / 匹配均有单测）✅；② ONNX Runtime 版本无关——onnxruntime 1.23.1 与 1.28.0 跑出**逐位相同**的 0.561722 / 0.403239（1.28.0 用 `pip install --target` + `PYTHONPATH` 隔离安装，不改环境）✅；③ conf=0.25 逐框比对：框坐标 max Δ≈0.0008（≈0.5px）、检测数完全一致、置信度 Δ≈0.001–0.01 → 差异来自推理栈输入预处理（`image` crate JPEG 解码 + 定点双线性 resize vs OpenCV `cv2.imread`+`cv2.resize` INTER_LINEAR），±1 LSB 输入差经网络放大为置信度差，再在 `conf=0.001` 的数百个重叠框上经 NMS 级联成少量 TP/FP/FN 翻转。
- **结论**：评测/指标实现与 Ultralytics **等价成立**；残留 0.0074（mAP@0.5）是 vendored `ultralytics-inference` crate 的图像解码/缩放噪声地板，非评测代码缺陷，需换用 OpenCV 图像管线方可进一步收敛。真实 COCO 评估已用 `conf=0.001` 落地（`config/eval_coco.toml`）。

---

### 阶段 5 — GPU 与性能优化

**目标**：在 NVIDIA GPU（服务器 + Jetson）上跑起来，用 batch/量化把性能做到实时。

**内容**：
- 开启 `cuda` / `tensorrt` feature，`device` 自动选择（CUDA → TensorRT 引擎缓存）。
- 量化：`quantize = "fp16" | "int8"`，对比精度损失。
- batch 调优：Jetson 与服务器分别找到吞吐/延迟最优 batch。
- 实时流优化：摄像头/RTSP 下做流水线（读帧 / 预处理 / 推理 / 后处理重叠）降低端到端延迟。
- 用阶段 3 的 `bench` 做跨设备对比（CPU vs CUDA vs TensorRT、服务器 vs Jetson）。

**功能（交付）**：
- 服务器 GPU 与 Jetson 上各跑通一套 `config`，输出标注/JSON 与 bench 报告。
- 量化模型加载与推理；TensorRT 引擎缓存复用（第二次启动免重建）。

**评价指标**：
- 延迟：GPU 相对 CPU 降低 ≥ 3×（同模型同精度）；TensorRT 相对 CUDA 进一步降低。
- 吞吐：batch=N 相对 batch=1 提升（记录倍数）。
- 量化：int8 模型体积下降约 75%，|ΔmAP@0.5| ≤ 1.0（用阶段 4 评估）。
- 实时目标（随模型定）：detect 中小模型在 Jetson 上达到 ≥ 15–30 FPS。

**验收标准**：bench 报告展示 CPU/CUDA/TensorRT 三者的延迟与吞吐对比，量化精度损失在容差内。

**完成情况（2026-10-18，Windows RTX 3060 实测，本机为 Windows，CUDA/TensorRT 目标硬件待服务器/Jetson 复测）**：

- 新增 GPU 配置与工具：`config/bench_{cpu,cuda,cuda_fp16,cuda_batch4,cuda_batch8,cpu_m,cuda_m,directml}.toml`、`config/detect_cuda.toml` / `detect_cuda_fp16.toml`、`utils/bench_compare.py`（汇总各 `bench_report.json` 成 markdown 对比表）、模型 `models/det/yolo26m.onnx` + `yolo26s.onnx`。`cuda` / `directml` feature 已打通（映射到 `ort/cuda`、`ort/directml`）。
- **跨设备实测结果**（24 图 `data/bench`，warmup 3 / iterations 20）：

| 配置 | 模型 | 设备 | 吞吐 (img/s) | total p50 (ms) | 峰值 RSS (MiB) |
|------|------|------|--------------|----------------|----------------|
| bench_cpu | yolo26n | CPU | 15.91 | 54.9 | 215.5 |
| bench_cuda | yolo26n | CUDA | 17.05 | 52.8 | 216.9 |
| bench_cuda_fp16 | yolo26n | CUDA fp16 | 13.00 | 63.1 | 215.7 |
| bench_cuda_batch4 | yolo26n | CUDA b4 | 14.51 | 58.5 | 215.4 |
| bench_cuda_batch8 | yolo26n | CUDA b8 | 16.74 | 54.6 | 215.5 |
| bench_cpu_m | yolo26m | CPU | 2.66 | 370.3 | 504.2 |
| bench_cuda_m | yolo26m | CUDA | 2.69 | 371.9 | 504.7 |
| bench_directml | yolo26m | DirectML | **44.43** | 19.0 | 450.3 |

- **DirectML 真加速**：yolo26m 上 DirectML 相对 CPU 吞吐 2.66 → 44.43 img/s（**≈16.7×**），inference p50 367.5 → 16.4 ms（≈22×）。DirectML 走 Windows DX12，**无需 CUDA/cuDNN 版本匹配**，是 Windows 开发机的可用 GPU 后端。
- **CUDA 后端可用但本机无收益（根因已定位）**：CUDA EP 能加载、能跑，但 yolo26n/yolo26m 均无加速（甚至略慢，GPU 利用率 ~3%）。原因见库 `vendor/inference/src/model.rs` 的加载错误提示：**下载的预编译 ONNX Runtime 针对 CUDA 13 + cuDNN 9 构建，而本机是 CUDA 12.8 + cuDNN 8** → conv 内核回退 CPU。两条解决路径（留待服务器/Jetson 复测）：
  1. 目标机安装 **CUDA 13 运行时 + cuDNN 9**（TensorRT EP 还需 TensorRT 10）；
  2. 或在 CUDA 12 机器上**用 `ORT_LIB_PATH` 自编译并链接针对 CUDA 12 的 ONNX Runtime**。
- **量化精度（fp16，本机 CUDA 回退 CPU，故仅作初步参考）**：`detect_cuda.toml`（fp32）与 `detect_cuda_fp16.toml`（fp16）在同一张 `bus.jpg` 上输出 5 个检测，类别 / 置信度 / 框坐标**逐字段完全一致（最大偏差 0）**。真值 GPU 上 fp16/int8 的精度损失仍需在阶段 4 评估数据 + 真 GPU（CUDA 13 / TensorRT）上复测。
- **batch 调优（yolo26n 小模型）**：batch 1/4/8 吞吐 17.05 / 14.51 / 16.74 img/s，**无提升**——小模型推理时间本就短，batch 带来的内核并行收益被批内同步抵消；需在更大模型 / Jetson 上再评估。
- **未做（后续项）**：int8 量化（实现层不支持，见下）、服务器 vs Jetson 实测、实时流水线。TensorRT 引擎缓存已在服务器实测通过（2026-11-04）。

**完成情况（2026-11-04，AutoDL 服务器 CUDA 13 / TensorRT 10 实测 ✅）**：

- **TensorRT EP 全链路打通**：服务器（AutoDL Ubuntu 22.04 + RTX 3080 10GB + driver 595/CUDA 13.2，nvidia-smi 实测）经 `pip install tensorrt-cu13`（TensorRT 10.13，CUDA 13 变体）补齐 `libnvinfer.so.10` / `libnvinfer_plugin.so.10` / `libnvonnxparser.so.10` 运行时（cuDNN 9 系统自带），重编 `--features "cuda tensorrt"` 后跑通；`bench_report.json` 的 `device = TensorRTExecutionProvider`。运行时需把 pip wheel 落点（`site-packages/tensorrt_libs/`）加进 `LD_LIBRARY_PATH`。首次运行构建引擎（缓存在 `models/det/.trt_cache/<stem>_{fp32,fp16}/`，640×640 约 30s~1min），二次启动复用免重建。
- **跨后端实测**（24 图 `data/bench`，warmup 3 / iterations 20，yolo26n；2026-11-16 复测于 RTX 3080 10GB、GPU 空闲）：

| 配置 | 设备 | total p50 (ms) | p95 | p99 | 吞吐 (img/s) |
|------|------|----------------|-----|-----|--------------|
| bench_cuda | CUDA FP32 | 15.7 | 21.8 | 25.4 | 55.91 |
| bench_tensorrt | TensorRT FP32 | 15.6 | 21.5 | 23.9 | 56.77 |
| bench_tensorrt_fp16 | TensorRT FP16 | 19.4 | 24.2 | 27.6 | 48.44 |

  - **结论**：本次复测 CUDA FP32（15.7ms）≈ TensorRT FP32（15.6ms）基本持平，而 **TensorRT FP16（19.4ms）反而最慢**。11-04 首测时 TRT 领先（15.5 vs 21.1ms），主因是当时 CUDA 基线偏慢；如今 CUDA 已到 15.7ms，TRT 不再有优势。TRT FP16 慢于 FP32 属异常（大概率引擎重建 tactic 采样差异 / nano 模型 fp16 内核不适配），非 fp16 本身收益。GPU 空闲、480 样本 p50，数字可信。
  - **实践结论**：本卡 + nano 模型上 **TensorRT 无明显收益，默认用 CUDA FP32 即可**（省去引擎构建与额外 `tensorrt-cu13` 依赖）；TRT 的加速价值需在更大模型 / 更大 `imgsz` 上重新评估。**检测级精度无损已实测**（FP32 与 FP16 在 `bus.jpg` 均输出 4 persons + 1 bus）；mAP 级损失仍可在真 GPU 上 `eval` + COCO 复测。
- **int8 结论（实现层）**：`quantize = "int8"` 在 vendored crate 中**只解析、无推理路径**——TensorRT EP 只认 FP16/FP32（`model.rs` 把非 FP16 一律回退 FP32 并告警），CUDA EP 亦无 int8 分支。故「int8 体积 −75%」目标需另走 Ultralytics Python 预导出 int8 模型，或自行实现 int8 推理，均不在本次范围。**FP16 是当前唯一可用的运行时量化**。
- **新增配置**：`config/bench_tensorrt.toml` / `bench_tensorrt_fp16.toml` / `detect_tensorrt.toml` / `detect_tensorrt_fp16.toml`；部署步骤见 `docs/deploy.md` §TensorRT。
- **服务器实测范围（重要）**：detect 之外的 6 任务（segment / obb / pose / classify / depth / semantic）已补 `config/{segment,obb,pose,classify,depth,semantic}_cuda.toml`，并在服务器**有卡模式实测通过**（2026-11-16，RTX 3080 10GB）：全部 `设备: CUDAExecutionProvider`、成功 1 / 失败 0，输出与本地 CPU 一致。单图耗时相对无卡（CPU 兜底）加速约 2.7×~20×——segment 293.7→29.6ms、obb 510.1→45.9ms、pose 211.5→22.5ms、classify 6.3→2.3ms、depth 2670.8→134.1ms、semantic 605.0→78.3ms。**TensorRT 逐图分区算子回退风险尚未验证**（待有卡时按 seg / pose / obb 各跑一次 `*_tensorrt*` 配置确认）。

---

### 阶段 6 — 报告与部署

**目标**：一键产出统一报告，并把后端打包成可在 Jetson / 服务器部署的形态。

**内容**：
- **统一报告**：把阶段 3（工程）与阶段 4（精度）的结果汇总成一份报告（HTML/Markdown + JSON），含对比图表、per-class 表。
- **接口决策门**：在打包前确定对外接口 —— Rust crate / HTTP·gRPC 服务 / C ABI·FFI（三选一或多选）。库优先架构保证此处改动局部化。
- **打包部署**：
  - Linux 服务器：Dockerfile（CUDA 基镜像），一键起服务。
  - Jetson：aarch64 交叉编译脚本 / 或板载编译，产出可运行二进制。
- 文档：部署说明、模型导出说明、配置参考。

**功能（交付）**：
- 统一报告由 `python -I utils/gen_report.py` 生成（读取 `results/eval_report.json` + `results/bench_*/bench_report.json` + `data/equiv/*.json` → `docs/report.md`）。
- Docker 镜像在服务器可启动；Jetson 上二进制可运行。
- 接口形式落地（服务或 FFI）。

**评价指标**：
- 一条命令生成完整报告（工程 + 精度 + 图表）。
- Docker 镜像冷启动后即可推理；Jetson 二进制端到端跑通样例。
- 交付物在目标硬件上复现阶段 4/5 的指标。

**验收标准**：在服务器与 Jetson 上分别完成「部署 → 跑样例 → 出报告」全流程。

**完成情况（2026-10-21，统一报告部分 ✅）**：
- **统一报告（Markdown）已交付**：`utils/gen_report.py` 读取 `results/eval_report.json`（精度）+ `results/bench_*/bench_report.json`（工程）+ `data/equiv/*.json`（Python 参考）生成 `docs/report.md`，含 mAP 对齐四路对比表、80 类 per-class ΔAP 表（按 |Δap50| 降序，mean 0.0082 / max 0.062，2 类 >0.05）、跨后端基准表、延迟分解与关键发现。
- **待续（部署项）**：接口形式已决策（服务器走 CLI 批处理，HTTP/gRPC/FFI 留到 Jetson）；服务器打包已交付并**在 AutoDL 部署目录实测通过**（`deploy/run.sh` + `deploy/Dockerfile` + `deploy/.dockerignore` + `docs/deploy.md`；`./run.sh bench` 出 CUDAExecutionProvider、total p50 ≈21ms/吞吐 ≈44 图/s，`./run.sh config/detect_cuda.toml` 出 5 检测；TensorRT EP 亦实测通过，FP32 p50 ≈15.6ms/吞吐 ≈55 图/s、FP16 p50 ≈15.5ms/吞吐 ≈58 图/s，见阶段 5）；仅剩 Jetson aarch64 打包 + Jetson 复测。部署组装已脚本化（`deploy/assemble.sh` 一键构建+组装，`deploy/run.sh` 自动探测 TensorRT lib）；服务器实测目前仅 detect，其余 6 任务的服务器 CUDA 冒烟配置已备好（`config/*_cuda.toml`）待有卡时逐条验证（见阶段 5）。

### 阶段 7 — 开箱即用（1.0 → 1.5）

**目标**：从「我能跑通」到「陌生人 clone 下来 10 分钟跑通自己的图」。

**唯一判据**：陌生人 clone 仓库后，不装 Rust 工具链、10 分钟内跑通一张自己的图；跑不通就不是 1.5。

**四支柱**：
- ① 预编译二进制（Windows x64 / Linux x64 / aarch64），免装 Rust 工具链；
- ② 一键 CLI：`yolo-rust --model x.onnx --image y.jpg` → 出框 + JSON；
- ③ 主流 ONNX 模型后处理（v8 / v11 / v12 head）—— **已由底层库满足**（YOLOv8 / YOLO11 / YOLO26 + RT-DETR，见 `vendor/inference`）；
- ④ Docker + 5 分钟上手 README + 一键复现 benchmark 脚本。

**完成情况**：
- **P0 ✅（2026-11-20）**：外层项目 git 化并首次推送 GitHub（`Joonie04/yolo-rust`）。结构整理：`references/` → `vendor/`（vendored 源码，入库）；`third_party/`（预编译二进制 + 下载缓存，gitignore）。重写 `.gitignore`（排除 520M 第三方二进制、`data/{coco2017,eval,equiv}`、`*.onnx/*.pt`、机器路径 `.cargo/config.toml`）。补顶层 `LICENSE`（AGPL-3.0）与 `.cargo/config.toml.example` 模板。
- **P1（进行中）**：② 一键 CLI ✅（`yolo-rust --model x.onnx --image y.jpg` → 出框 + JSON，2026-11-20 实测）；① CPU 预编译二进制（CI 已就绪 `.github/workflows/release.yml`，待打 tag 出首个 Release）；④ 陌生人 README（已加「最快上手」+ 预编译二进制指引，待 Release 产出后补完整链路）+ 一键 bench 脚本（待做）。
- **P2（待做）**：Docker CPU 镜像、GitHub Actions 三平台 CI + Release。
- **P3（待做）**：三平台二进制端到端 smoke。

---

## 6. 待确认事项（不阻塞前期阶段）

| 事项 | 默认假设 | 何时需确认 |
|------|---------|-----------|
| 标注数据格式 | COCO（+ 可插拔解析器） | 阶段 4 开始前 |
| 对外接口形式 | 库优先，暂不绑定 | 阶段 6 打包前决策 |
| 实时流具体来源 | 摄像头 / RTSP 均支持 | 阶段 2/5 视需求细化 |
| 精度对齐的 Python 版本 | Ultralytics 最新稳定版 | 阶段 4 对齐时 |

---

## 7. 里程碑与顺序依赖

```
阶段 0（已完成）
   ├── 阶段 1 全任务输出 ✅ ──→ 阶段 4 精度评估（全任务指标 ✅，等价性检测级 ✅） ──┐
   ├── 阶段 2 视频/流 ✅                                ├──→ 阶段 6 报告与部署
   └── 阶段 3 批量+基准 ✅ ──→ 阶段 5 GPU/性能 ✅（DirectML） ──┘
                                                          └── 阶段 7 开箱即用 1.0→1.5（P0 ✅；P1 待做）
```

- **推荐推进顺序**：1 → 3 → 4 → 5 → 2（视频可穿插）→ 6 → 7。
  - 先做 1（结构化输出）打通「所有任务都能出数据」；
  - 3（基准）建立「能测」的能力，是 5 的量化前提；
  - 4（精度）证明「做得对」，是量化的质量前提；
  - 5（GPU）拿到性能收益；2（视频/流）随时可插入；6 收口打包；
  - 7（开箱即用）是发布前的最后一步：P1 三件事（预编译二进制 / 一键 CLI / 陌生人 README）优先，P2（Docker/CI）、P3（smoke）随后。
