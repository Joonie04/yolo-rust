# CPU 推理镜像（多阶段：builder 编译 → runtime 只含二进制 + 官方 ORT 动态库）。
#
# 用法：
#   docker build -t yolo-rust .
#   docker run --rm -v "$PWD:/data" yolo-rust --model yolo26n.onnx --image data/bus.jpg
#
# 预构建镜像（推 tag 时自动发到 GHCR）：见 .github/workflows/release.yml。
#
# 说明：
#   - builder 用 ubuntu:22.04（glibc 2.35）+ 官方 ORT 动态链接（manylinux_2_28 / glibc 2.28），
#     绕开 pyke 静态 ORT 需 glibc 2.38+ / GCC 13 的问题（见 docs/roadmap.md 阶段 7）。
#   - runtime 装 libssl3（下载模型需 TLS）+ ca-certificates，并把 libonnxruntime*.so 放进系统库路径。

FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY . .
RUN curl -fL --retry 3 -o ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz" \
    && tar xzf ort.tgz \
    && ORT_LIB_LOCATION=/app/onnxruntime-linux-x64-1.28.0/lib \
       ORT_PREFER_DYNAMIC_LINK=1 \
       cargo build --release --locked

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/yolo-rust /usr/local/bin/yolo-rust
COPY --from=builder /app/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime*.so* /usr/local/lib/
RUN ldconfig
WORKDIR /data
ENTRYPOINT ["yolo-rust"]
