# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

配置驱动的 YOLO 推理项目（package/bin 名 `yolo-rust`，lib crate 名 `yolo_rust`，edition 2024 / MSRV 1.92）。它是对 vendored 底层库 **`ultralytics-inference`**（path 依赖，位于 `vendor/inference`，基于 ONNX Runtime）的一层薄封装：业务层只写 TOML 配置（`config/*.toml`），推理逻辑、模型、任务全交给底层库。覆盖 7 类任务（detect / segment / semantic / depth / classify / pose / obb）× 4 类输入（image / directory / glob / video+webcam+stream）。

## 硬性约定

- **项目根 `.claude` 文件是唯一站立指令**：「每次任务结束后，记得更新开发文档」。改完代码必须同步更新 `docs/roadmap.md`（阶段完成情况）、`docs/deploy.md`、`docs/plan.md` 等对应文档。
- **外层项目现已 git 化**（含 `.git`，远程 `github.com/Joonie04/yolo-rust`）。
- **`vendor/inference/` 是上游 `ultralytics/inference` 的 vendored 副本**，带自己的 `AGENTS.md`（Delete > Replace > Add、git worktree、绝不 push main）。优先在外层解决；确需改动底层时遵守其 AGENTS.md，且不要在 vendored 副本内 push。

## 命令

```bash
# 三个子命令（CLI 入口在 src/main.rs）
cargo run -- config/config.toml           # 推理（单图/目录/glob/视频/流）
cargo run -- bench config/bench.toml      # 工程基准 → <bench.dir>/bench_report.json
cargo run -- eval config/eval.toml        # 精度评估 → <eval.dir>/eval_report.json

# GPU/后端 feature（在根 Cargo.toml 里透传给 ultralytics-inference）
cargo run --release --features cuda -- config/config.toml          # NVIDIA CUDA
cargo run --release --features "cuda tensorrt" -- bench config/bench_tensorrt_fp16.toml
cargo run --release --features directml -- config/config.toml      # Windows GPU
cargo run --release --features video -- config/video.toml          # 视频/流（需 FFmpeg + libclang）

# 代码卫生
cargo fmt
cargo clippy --all-targets
```

- **外层 crate 没有单元/集成测试**（`src/` 下无 `#[cfg(test)]`，无 `tests/`）。有效冒烟测试 = 跑一个最小 config（如 `config/detect.toml` 出检测、`config/bench.toml` 出报告）。底层库自身的测试/lint 命令见 `vendor/inference/AGENTS.md`。
- `default` feature 只有 `annotate`。视频推理必须 `--features video`（`engine.rs` 会在运行时拦截并提示）。
- 首次构建会联网下载 ONNX Runtime 二进制（走 ort 的 `download-binaries`）。

## 架构（大图）

数据流：**config → `src/config.rs` 加载校验 → 映射为库 `InferenceConfig` → `src/engine.rs` 驱动 `YOLOModel` → `src/source.rs` 遍历输入 → `src/output.rs` 落盘 JSON + 标注图**。

`src/` 八个模块，职责单一：

| 模块 | 职责 |
|------|------|
| `main.rs` | 极薄 CLI：解析子命令（`bench`/`eval`/缺省=推理），转发到 lib 三个入口 |
| `lib.rs` | 三个 pub 入口 `run` / `bench` / `eval` |
| `config.rs` | 三套配置 `AppConfig` / `BenchConfig` / `EvalConfig`，`deny_unknown_fields` + 语义校验；`InferenceSettings::to_inference_config()` 把可选项映射为库的 `InferenceConfig` |
| `engine.rs` | `InferenceEngine::run`：加载模型 → 开源 → 逐张 `predict_image` → 单张失败不中断（记入 `failures`） |
| `source.rs` | 配置的 `source.type` 字符串 → 库 `Source` 枚举 → `SourceIterator` |
| `output.rs` | JSON「公共信封 + 任务载荷」（detections/masks/keypoints/obb/probs/semantic/depth），坐标归一化到 [0,1]；标注图/视频经 `SaveResults` 编码线程落盘；输出目录 `results/<task>/<unix秒时间戳>/` |
| `bench.rs` | 整批图先载入内存（排除磁盘 I/O），预热 → 计时 → `sysinfo` 采样峰值 RSS；`batch>1` 走 `predict_batch` |
| `eval.rs` | 按 `model.task()` 自动分派：detect=mAP@0.5:0.95（101 点插值）、classify=top1/top5、pose=OKS mAP+PCK@0.5、segment=掩码 IoU、semantic=mIoU、obb=旋转 IoU；预测与真值按**类别名称**对齐 |

**关键约束**：config 里的 `model.path` / `source.path` / `[output].dir` 都是**相对工作目录**，所以二进制必须从仓库根（或部署根）启动。这正是 `deploy/run.sh` 先 `cd` 到自身目录再 `exec` 的原因。

## GPU / ORT 要点（已在服务器实测）

- feature 链：`yolo-rust --features cuda tensorrt` → `ultralytics-inference/{cuda,tensorrt}` → `ort/{cuda,tensorrt}`。没有 `nvidia` 别名，直接写 `cuda` 或 `cuda tensorrt`。
- 构建期用 `ORT_LIB_LOCATION=<官方 gpu_cuda13 的 lib/>` + `ORT_PREFER_DYNAMIC_LINK=1` 可跳过 ort 下载、动态链 `.so`（Linux 服务器已验证；官方包需 glibc 2.28，Ubuntu 22.04 可用）。
- 运行期 `LD_LIBRARY_PATH` 要含 `onnxruntime/lib` 与（TensorRT 时）pip `tensorrt-cu13` 的 `site-packages/tensorrt_libs/`——后者 `deploy/run.sh` 已自动探测。
- **`quantize = "int8"` 在 vendored crate 里只解析、无推理路径**，TensorRT EP 会回退 FP32；唯一真实的运行时量化是 **fp16**。device 写法 `tensorrt:0` / `cuda:0` 由库的 `Device::from_str` 解析。
- Windows 开发机：`video` feature 的 FFmpeg 已 vendor 到 `third_party/ffmpeg/`，路径由 `.cargo/config.toml` 的 `[env]` 注入（`FFMPEG_DIR`、`LIBCLANG_PATH`——后者是机器相关路径 `D:/anaconda3/Library/bin`）。Cargo `[env]` 不支持按 `cfg` 作用域（见该文件注释）。

## 部署

- **`deploy/assemble.sh`**：一键「构建 + 组装」自包含部署目录（`--tensorrt` / `--cuda` / `--skip-build`，`DEP`/`ORT_LIB` 可用环境变量覆盖），自动排除 `data/` 里 `coco2017/eval/equiv` 大目录、删除 `.pt` 只留 `.onnx`。
- **`deploy/run.sh`**：部署目录入口，固化 `LD_LIBRARY_PATH`（含 TensorRT 自动探测）并从自身目录启动。
- 完整部署/验证清单见 `docs/deploy.md`；分阶段路线图与完成状态见 `docs/roadmap.md`。

## 其它

- `utils/`：`export_onnx.py`（.pt→.onnx）、`make_synthetic_coco.py` 等生成合成 GT、`bench_compare.py`（汇总 bench 报告）、`gen_report.py`（生成 `docs/report.md`）。跑 Python 脚本请加 `python -I`（仓库内脚本与数据互不污染）。
- `examples/`：`make_segment_gt` / `make_semantic_gt` / `make_obb_gt` 生成合成评估真值。
- 模型放 `models/<task>/`（仅 `.onnx` 参与部署，`.pt` 已被 `.gitignore` 排除）；评估大数据在 `data/coco2017`、`data/eval`、`data/equiv`。
