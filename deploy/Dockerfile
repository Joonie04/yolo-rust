# 服务器 GPU 运行时镜像（AutoDL：Ubuntu 22.04 + RTX 3080（10GB），CUDA 13）。
#
# 说明：本镜像只「打包」，不在容器内从源码编译（避免国内网络重复拉 crates / 官方 ORT 包）。
# 二进制 + ORT 动态库需预先在服务器上构建好，并组装到本目录（见 docs/deploy.md §5）。
#
# 构建（在部署根目录内，构建上下文 = 本目录）：
#   docker build -t yolo-rust:gpu .
# 运行：
#   docker run --gpus all \
#     -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#     yolo-rust:gpu config/detect_cuda.toml

# 13.2 运行时自带 CUDA 13 运行时库（cudart/cublas/cublasLt），满足 ORT gpu_cuda13 的运行期依赖。
# 若该 tag 不存在，换 13.1-runtime-ubuntu22.04 / 13.0-runtime-ubuntu22.04（driver 13.2 向后兼容 ≤13.2）。
FROM nvidia/cuda:13.2-runtime-ubuntu22.04

WORKDIR /app

# 部署资产（二进制 / ORT 库 / 模型 / 配置 / 入口脚本）。
COPY yolo-rust        ./
COPY onnxruntime/lib  ./onnxruntime/lib
COPY models           ./models
COPY config           ./config
COPY run.sh           ./

# run.sh 需可执行（Windows 下拷入的文件无 +x）。
RUN chmod +x /app/run.sh

# 运行时依赖：ORT 动态库（CUDA 13 运行时库由基础镜像提供）。
ENV LD_LIBRARY_PATH=/app/onnxruntime/lib

# data/ 与 results/ 在运行时挂载，不打进镜像。
ENTRYPOINT ["/app/run.sh"]
