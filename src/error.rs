//! 外层项目统一错误类型。

use thiserror::Error;

/// 外层项目错误。
#[derive(Debug, Error)]
pub enum AppError {
    /// 读取配置文件失败。
    #[error("读取配置文件 `{path}` 失败: {source}")]
    ConfigIo {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// 配置文件解析失败。
    #[error("配置文件解析失败: {0}")]
    ConfigParse(#[from] toml::de::Error),
    /// 配置内容非法（语义校验失败）。
    #[error("配置错误: {0}")]
    Config(String),
    /// 底层推理库错误。
    #[error("{0}")]
    Inference(#[from] ultralytics_inference::InferenceError),
    /// JSON 序列化失败。
    #[error("JSON 序列化失败: {0}")]
    Json(#[from] serde_json::Error),
    /// 普通 IO 错误。
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

/// 外层项目 Result 别名。
pub type Result<T> = std::result::Result<T, AppError>;
