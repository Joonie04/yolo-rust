# CPU 推理镜像（多阶段：builder 编译 → runtime 只含二进制）。
#
# 用法：
#   docker build -t yolo-rust .
#   docker run --rm -v "$PWD:/data" yolo-rust --model yolo26n.onnx --image data/bus.jpg
#
# 预构建镜像（推 tag 时自动发到 GHCR）：见 .github/workflows/release.yml。
#
# 说明：
#   - builder/runtime 用 ubuntu:24.04（glibc 2.39）：pyke 静态 ORT 用 GCC 13 + glibc 2.38 编译，
#     需 glibc ≥ 2.38 才能链接/运行。
#   - runtime 装 libssl3（下载模型需 TLS）+ ca-certificates。二进制为单文件（静态链接 ORT）。

FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY . .
RUN cargo build --release --locked

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/yolo-rust /usr/local/bin/yolo-rust
WORKDIR /data
ENTRYPOINT ["yolo-rust"]
