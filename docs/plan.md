# YOLO 推理项目设计文档

> 基于 `vendor/inference`（`ultralytics-inference` v0.0.51）构建的外层 YOLO 推理项目。
> 核心目标：**通过配置文件调整要推理的信息**（模型、输入源、推理参数、输出形态），业务代码与推理逻辑解耦。

---

## 1. 项目概述

- 底层库：`ultralytics-inference`（ONNX Runtime，Rust，AGPL-3.0）
- 库公开 API（本项目的使用面）：
  - `YOLOModel::load(path)` / `YOLOModel::load_with_config(path, config)`
  - `model.predict(path)`、`model.predict_image(&img, path)`、`model.predict_batch(&[img], &[path])`
  - `InferenceConfig::new().with_*()` 构建器
  - `Device`、`Quantization`、`Task`（均可 `str::parse`）
  - `Source` / `SourceIterator`（图片、目录、glob、视频、摄像头、流）
  - `Results`（`boxes` / `masks` / `keypoints` / `probs` / `obb` / `semantic_mask` / `depth` / `speed` / `names`）
- 关键约束：
  - 库只支持 **ONNX** 模型，不支持 `.pt`（PyTorch 权重需先导出）
  - 库 MSRV = Rust **1.92**，edition **2024**
  - 首次构建会联网下载 ONNX Runtime 二进制
  - 库默认 feature = `annotate` + `visualize`；Windows GPU 走 `directml`

---

## 2. 现状问题与补充

| # | 问题 | 处理方式 |
|---|------|----------|
| 1 | `models/` 全是 `.pt`，库只能跑 `.onnx` | 提供 `utils/export_onnx.py` 批量导出；配置指向 `.onnx` |
| 2 | 根目录 `main.rs` 与 `src/` 布局冲突 | 删除根 `main.rs`，统一使用 `src/main.rs` |
| 3 | `utils/` 放 Rust 源码不标准 | Rust 源码全部进 `src/`，`utils/` 仅放脚本 |
| 4 | 缺 `Cargo.toml` / `.gitignore` / 示例配置 | 补齐 |
| 5 | `results/` 为生成物 | 加入 `.gitignore` |
| 6 | 库为 AGPL-3.0 | 分发时需 AGPL 兼容或 Ultralytics 企业授权 |
| 7 | `vendor/inference` 自带 `[workspace]` | 用 path 依赖引用，不嵌套进外层 workspace |
| 8 | 工具链版本较新（1.92） | 需安装较新的 Rust 工具链 |

---

## 3. 目标目录结构

```
yolo_rust/
├── Cargo.toml                  # 外层 crate（path 依赖 vendor/inference）
├── .gitignore                  # target/ results/ *.onnx 等
├── README.md
├── config/
│   ├── config.toml             # 默认配置（示例，可改）
│   ├── detect.toml             # 检测示例
│   ├── segment.toml            # 分割示例
│   └── pose.toml               # 姿态示例
├── data/                       # 输入：图片 / 视频（按任务建子目录）
├── models/                     # ONNX 模型（.pt 导出后放这里，按任务分 det/seg/pose/...）
├── results/                    # 生成物：标注图 + JSON + 日志（gitignore）
├── docs/                       # 架构、用法说明
├── vendor/
│   └── inference/              # 引用的库（保持不动）
├── utils/
│   └── export_onnx.py          # 批量 .pt -> .onnx 导出脚本
└── src/
    ├── lib.rs                  # 对外入口 run() + 模块声明
    ├── main.rs                 # 薄二进制：解析参数 -> lib::run
    ├── config.rs               # AppConfig 结构 + TOML 反序列化 + 校验 + 映射到 InferenceConfig
    ├── engine.rs               # InferenceEngine：加载模型、跑推理
    ├── source.rs               # 把 [source] 解析成 Source，遍历图像
    ├── output.rs               # 输出：控制台摘要 / JSON / 标注图落盘
    └── error.rs                # AppError（thiserror）
```

> `lib.rs` + `main.rs` 分离是标准做法，便于以后加集成测试；若只想要单一二进制，也可合并为纯 `main.rs` + 子模块。

---

## 4. 配置设计（核心）

采用 **TOML**（`serde` + `toml` 直接反序列化，Rust 生态最自然）。

### 4.1 配置示例 `config/config.toml`

```toml
[model]
path = "models/det/yolo26n.onnx"   # 必填：ONNX 路径；也接受 "yolo26n.onnx"（库会自动下载）
                                   # task 不需要填 —— 库会从 ONNX 元数据自动识别

[source]
type = "image"                     # image | directory | glob | video | webcam | stream | list
path = "data/bus.jpg"              # 图片路径 / 目录 / glob 模式 / 视频路径 / rtsp / 摄像头索引 / 路径列表

[inference]
device = "cpu"                     # 缺省 = 自动选最优 provider；可选 cpu|cuda:0|tensorrt:0|directml:0|coreml|intel:gpu|rocm:0|xnnpack
conf = 0.25                        # 置信度阈值
iou = 0.7                          # NMS IoU
max_det = 300                      # 每图最大检测数
# imgsz = [640, 640]               # 可选；缺省用模型元数据
# batch = 1                        # 可选
threads = 0                        # 0 = 全部核心
# quantize = "fp16"                # 可选：8/16/32/int8/fp16/fp32/w8a8/w16a16/w32a32/w8a16/w8a32
rect = true                        # 矩形推理（最小 padding）
# classes = [0, 2]                 # 可选：只保留这些类别

[output]
save_annotated = true              # 保存标注图
save_json = true                   # 保存结构化 JSON
dir = "results"                    # 输出根目录（自动追加 runs/<task>/<时间戳>/）
# 视频相关字段（save_frames / show）留待视频阶段再添加

[logging]
verbose = true
```

### 4.2 配置项 → 库 API 映射

在 `config.rs` 中做一次转换，把 `AppConfig` 变成 `InferenceConfig`：

| 配置项 | 库调用 |
|--------|--------|
| `conf` | `with_confidence(f32)` |
| `iou` | `with_iou(f32)` |
| `max_det` | `with_max_det(usize)` |
| `imgsz=[h,w]` | `with_imgsz(h, w)` |
| `batch` | `with_batch(usize)` |
| `threads` | `with_threads(usize)` |
| `quantize="fp16"` | `str::parse::<Quantization>()` → `with_quantize` |
| `device="cuda:0"` | `str::parse::<Device>()` → `with_device` |
| `rect` | `with_rect(bool)` |
| `classes=[..]` | `with_classes(Vec<usize>)` |

> 说明：`save`/`save_frames` 只作用于库自带的 CLI，`YOLOModel::predict` 路径并不读取它们；
> 标注图/JSON 的落盘由外层 `output.rs` 自行处理，不映射进 `InferenceConfig`。
> 各参数中仅 `batch` 会真正影响推理（`predict_internal` 固定 batch 分块与 warmup）。

- 加载模型：`YOLOModel::load_with_config(path, config)`
- 单图：`model.predict(path)`
- 目录 / glob / 视频：`SourceIterator::new(Source::from(path))` 遍历 → `predict_image` / `predict_batch`

---

## 5. 各模块职责

| 模块 | 职责 |
|------|------|
| `config.rs` | `AppConfig`（`#[derive(Deserialize)]`）含 `ModelConfig` / `SourceConfig` / `InferenceSettings` / `OutputConfig` / `LoggingConfig`；`AppConfig::load(path)`；`InferenceSettings::to_inference_config()` |
| `source.rs` | 用 `Source::from(path_str)` 自动识别 image/dir/glob/video/webcam/stream；`SourceIterator` 统一遍历，产出 `(DynamicImage, SourceMeta)` |
| `engine.rs` | `InferenceEngine::new(&AppConfig)` 建模型；`run()` 遍历 source，收集 `Vec<Results>`，交给 `output.rs` |
| `output.rs` | 控制台打印 `detection_summary()` + `speed.total()`；JSON 用 `result.summary(normalize=true)` 转 `serde_json::Value`；标注图用 `result.save(path)` |
| `error.rs` | `AppError`（thiserror），包装 `InferenceError` / `toml::de::Error` / `std::io::Error` |
| `main.rs` | 读 `args[1]`（缺省 `config/config.toml`）→ `yolo_rust::run(path)`，出错打印并 `exit(1)` |

---

## 6. Cargo.toml 设计

```toml
[package]
name = "yolo_rust"
version = "0.1.0"
edition = "2024"
rust-version = "1.92"

[dependencies]
ultralytics-inference = { path = "vendor/inference", default-features = false }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
serde_json = "1"
thiserror = "2"

# 把库的 feature 透传出去，方便按需开 GPU / 视频
[features]
default = ["annotate"]
annotate   = ["ultralytics-inference/annotate"]
visualize  = ["ultralytics-inference/visualize"]
video      = ["ultralytics-inference/video"]
cuda       = ["ultralytics-inference/cuda"]
tensorrt   = ["ultralytics-inference/tensorrt"]
directml   = ["ultralytics-inference/directml"]
openvino   = ["ultralytics-inference/openvino"]
coreml     = ["ultralytics-inference/coreml"]
```

> Windows GPU：`cargo run --release --features directml -- config/config.toml`
> CPU 默认：`cargo run -- config/config.toml`

---

## 7. 运行流程

```
cargo run -- config/config.toml
        │
        ▼
main.rs ── 解析参数（配置路径，缺省 config/config.toml）
        │
        ▼
lib.rs::run()
        │
        ├─ config.rs   加载并校验 AppConfig
        ├─ engine.rs   构建 InferenceConfig -> YOLOModel::load_with_config
        ├─ source.rs   Source::from -> SourceIterator 遍历输入
        └─ output.rs   每个 Results：控制台摘要 / JSON / 标注图
```

---

## 8. 实施步骤

1. 建 `Cargo.toml` + `.gitignore` + 删除根 `main.rs`（迁到 `src/main.rs`）。
2. `src/error.rs` → `src/config.rs`（结构 + 反序列化 + 映射 + 校验）。
3. `src/source.rs` → `src/output.rs` → `src/engine.rs` → `src/lib.rs` + `src/main.rs`。
4. 写 `config/config.toml`（detect 示例）+ `utils/export_onnx.py`。
5. 跑通链路：导出 `yolo26n.pt` → ONNX，`cargo run -- config/config.toml`，验证控制台 + `results/` 输出。

`utils/export_onnx.py` 要点（需安装 `ultralytics`）：

```python
from ultralytics import YOLO
import sys
for p in sys.argv[1:]:
    YOLO(p).export(format="onnx")   # 在同目录生成同名 .onnx
```

---

## 9. 已确认决策

1. 配置格式：**TOML**。
2. 模型：使用 Ultralytics 原始权重，**由使用者自行导出 ONNX**（配置指向 `.onnx`）。
3. 输出形态：**先完成图像标注 + JSON**；视频处理后续再添加。
