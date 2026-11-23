# CPU 推理镜像（多阶段：builder 编译 → runtime 只含二进制）。
#
# 用法：
#   docker build -t yolo-rust .                          # 本机构建（需联网，首次 ~10min 下载 ORT + crates）
#   docker run --rm -v "$PWD:/data" yolo-rust --model yolo26n.onnx --image data/bus.jpg
#
# 预构建镜像（GitHub Actions 推 tag 时自动发布到 GHCR）：
#   docker run --rm -v "$PWD:/data" ghcr.io/<owner>/yolo-rust --model yolo26n.onnx --image data/bus.jpg
#
# 说明：runtime 装 libssl3（ort 的 tls-native 下载模型需要 TLS）+ ca-certificates。

FROM rust:1-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/yolo-rust /usr/local/bin/yolo-rust
WORKDIR /data
ENTRYPOINT ["yolo-rust"]
