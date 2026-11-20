//! 输入源解析与遍历。

use std::path::PathBuf;

use ultralytics_inference::{Source, SourceIterator};

use crate::config::SourceConfig;
use crate::error::{AppError, Result};

/// 把配置里的源描述解析为库的 `Source`。
pub fn resolve_source(config: &SourceConfig) -> Result<Source> {
    match config.source_type.as_str() {
        "image" => Ok(Source::Image(PathBuf::from(&config.path))),
        "directory" => Ok(Source::Directory(PathBuf::from(&config.path))),
        "glob" => Ok(Source::Glob(config.path.clone())),
        "video" => Ok(Source::Video(PathBuf::from(&config.path))),
        "webcam" => {
            let idx = config
                .path
                .parse::<u32>()
                .map_err(|_| AppError::Config(format!("webcam 索引非法: '{}'", config.path)))?;
            Ok(Source::Webcam(idx))
        }
        "stream" => Ok(Source::Stream(config.path.clone())),
        other => Err(AppError::Config(format!("不支持的 source.type: '{other}'"))),
    }
}

/// 打开源迭代器。
pub fn open_source(source: Source) -> Result<SourceIterator> {
    Ok(SourceIterator::new(source)?)
}
