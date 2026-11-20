# Linux 构建环境清单（Ubuntu VM，CPU 先行）

> 目的：在本地虚拟机里模拟 Linux 服务器，把 yolo_rust 的 **Linux 构建 + CPU 全链路** 跑通，
> 作为阶段 6「打包部署」的前置。GPU（CUDA/TensorRT）不在本清单范围（本地 VM 无 GPU 直通，见 `report.md` 第 5 节 / 另走 WSL2 或云 GPU VM）。
> 版本锚点：Rust 1.92（edition 2024）、ort `=2.0.0-rc.13`（ONNX Runtime 1.28.0）、ffmpeg-next `^9.0`。

---

## 0. 分阶段策略

| 阶段 | 范围 | 依赖 |
|------|------|------|
| A（必做） | 最小 CPU 验证：build + 单图 detect + bench | 仅 Rust 工具链 + 基础构建库 |
| B（可选） | 加 `video` feature（视频/流） | 系统 FFmpeg ≥7 + libclang |

先做 A，通了再做 B。video 是 4 类输入之一，但不是「能跑起来」的前置。

---

## 1. 虚拟机准备

- VMware Workstation / VirtualBox 装 **Ubuntu**（CPU 验证用 24.04 LTS 即可；若要走阶段 B 且不想折腾 FFmpeg 版本，见 §5）。
- 资源建议：CPU ≥ 4 核、内存 ≥ 8 GB、磁盘 ≥ 30 GB（ORT 下载 + `target/` + 可选 COCO 数据）。
- 网络：首次构建需访问 github 下载 ONNX Runtime Linux 库（国内网络可能慢/断，重跑 `cargo build` 即可）。

---

## 2. 基础工具链（VM 内执行）

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev curl git

# Rust（stable 已远超 1.92，直接装 stable 即可）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version        # 需 >= 1.92
```

> `libssl-dev` + `pkg-config` 是 ort 的 `tls-native` feature（经 native-tls/openssl-sys）所需的；
> 若报 openssl-sys 找不到，先确认这两个包装好。

---

## 3. 拉取项目

- 项目不是 git 仓库，直接拷目录进 VM。**推荐用白名单脚本自动生成上传目录**（只拷要的，结果约 284M）：
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File utils\stage_linux_upload.ps1
  # 需要复跑 COCO eval 时加 -WithEval $true（额外带 val 子集，约 +1.5G）
  ```
  生成 `D:\yolo_linux_upload\`，zip 后传上去即可（几分钟，而不是 31G / 14 小时）。
- 手动拷贝时**排除**（Linux 用不上或体积大）：
  - `target/`（Windows 构建产物，2.4G）
  - `third_party/`（gyan.dev 的 **Windows DLL**，290M）
  - `data/coco2017/train2017/`（COCO 训练集 ~18G，本项目只推理，**永不需要**）
  - `data/coco2017/test2017/`（COCO 测试集 ~6.5G，无标注，仅官方提交用）
  - `models/**/*.pt`（PyTorch 权重，Rust 只用 `.onnx`）
- 保留：`src/`、`config/`、`models/**/*.onnx`、`data/bus.jpg` 等样例、`vendor/`、`utils/`、`.cargo/`（脚本会自动重写为 Linux 版，剔除 Windows 的 `[env]`）。

---

## 4. 阶段 A：最小 CPU 验证（无 video）

```bash
cargo build --release        # default features = annotate
cargo test                   # 可选，跑单测
cargo run -- config/config.toml          # 单图 detect → results.json + 标注图
cargo run -- bench config/bench_cpu.toml # 工程基准 → bench_report.json
cargo run -- eval config/eval_coco.toml  # 仅当已拷 COCO 数据
```

- 首次 `cargo build` 会触发 ort `download-binaries` 下载 `onnxruntime-linux-x64-<ver>` 并 `copy-dylibs` 拷 `.so` 到 `target/`，需网络。
- 国内网络下载失败时**重跑 `cargo build`**（ort 是自管下载，非 curl，断点续传不适用）。

**通过标准**：build 成功 + 单图 detect 出 JSON + bench 出报告。

---

## 5. 阶段 B：加 `video` feature（可选）

ffmpeg-next `^9.0` 绑定 FFmpeg 7/8/9 系列；**Ubuntu 24.04 自带的 FFmpeg 6.1 太旧**，会版本不匹配。两条路：

- **方案 1（省事）**：用自带 FFmpeg ≥7 的发行版/源（Ubuntu 25.04、Debian 13，或第三方 PPA）。
- **方案 2**：装 dev 包 + clang：
  ```bash
  sudo apt install -y clang libclang-dev \
      libavcodec-dev libavformat-dev libavutil-dev \
      libavfilter-dev libswscale-dev libswresample-dev
  ```

`.cargo/config.toml` 里的 `FFMPEG_DIR` / `LIBCLANG_PATH` 是 Windows 专用（Cargo 的 `[env]` 无法按
平台限定，只能写顶层）；`utils/stage_linux_upload.ps1` 生成的 Linux 目录里 `.cargo/config.toml`
已被重写为不含 `[env]` 的干净版，**无需手动改**；装好 dev 包后直接构建即可。

```bash
cargo build --release --features video
cargo run --release --features video -- config/video.toml   # 处理 data/sample.mp4
```

---

## 6. Windows → Linux 移植点汇总

| 项 | Windows 现状 | Linux 处理 |
|----|-------------|-----------|
| `.cargo/config.toml` 的 `LIBCLANG_PATH` / `FFMPEG_DIR` | 顶层 `[env]`（Windows 专用） | staging 脚本重写为无 `[env]` 的干净版，**无需手改** |
| `build.rs` copy-dlls | 拷 `.dll` 到 exe 旁 | Linux 天然 no-op（无 `.dll`），**无需改代码** |
| `directml` feature | Windows DX12 GPU 后端 | Linux 不存在，用 `cpu`；GPU 走 `cuda`/`tensorrt` |
| `third_party/ffmpeg/` | gyan.dev Windows 共享库 | 不拷贝，换 `apt` 的 ffmpeg dev 包 |

---

## 7. 验证通过标准（Checklist）

- [ ] `cargo build --release` 成功（首次下载 ORT Linux 库）
- [ ] `cargo test` 通过（可选）
- [ ] 单图 `detect` 跑通，产出 `results.json` + 标注图
- [ ] `bench` 产出 `bench_report.json`（CPU 基线，供后续跨设备对比）
- [ ] （阶段 B）`video` 跑通 `data/sample.mp4`，产出标注 mp4 + 帧级 JSON

---

## 8. 下一步衔接

1. CPU 全链路通 → 写 **Dockerfile**（CUDA 基镜像，先 smoke test CPU 路径）。
2. GPU 真加速 → **WSL2**（本机 3060，CUDA 12 路线，自编译 ort）或 **云 GPU VM**（CUDA 13 + TensorRT 10，命中 ort 预编译目标）。⚠️ 预编译(pyke) ORT GPU 用 **glibc ≥2.38 + GCC ≥12** 编译，Ubuntu 22.04（glibc 2.35）直接链接会报 `undefined symbol: __isoc23_strtol` / `_M_replace_cold`；**已解决（22.04 可用）**：下载官方 `onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz`，`export ORT_LIB_LOCATION=<解压>/lib ORT_PREFER_DYNAMIC_LINK=1` 后直接 `cargo build --release --features cuda`（官方包 manylinux_2_28 / glibc 2.28，兼容 22.04，零重编译）。已在 AutoDL 22.04 + RTX 3080（10GB）跑通，结果见 `results/bench_cuda_server/bench_report.json`（CUDA13 EP，19.7ms/图）。
3. Jetson aarch64 → 板载编译或交叉编译（`aarch64-unknown-linux-gnu` + `acl`/`tensorrt` feature）。
