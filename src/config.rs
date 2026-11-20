//! 配置加载、校验，以及到库 `InferenceConfig` 的映射。

use std::path::Path;

use serde::Deserialize;
use ultralytics_inference::{Device, InferenceConfig, Quantization};

use crate::error::{AppError, Result};

/// 顶层配置。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    /// 模型配置。
    pub model: ModelConfig,
    /// 输入源配置。
    pub source: SourceConfig,
    /// 推理参数（可选，缺省走库默认值）。
    #[serde(default)]
    pub inference: InferenceSettings,
    /// 输出配置（可选）。
    #[serde(default)]
    pub output: OutputConfig,
    /// 日志配置（可选）。
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl AppConfig {
    /// 从文件加载并校验配置。
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| AppError::ConfigIo {
            path: path.display().to_string(),
            source,
        })?;
        let config: Self = toml::from_str(&text)?;
        config.validate()?;
        Ok(config)
    }

    /// 语义校验（TOML 结构之外的约束）。
    fn validate(&self) -> Result<()> {
        if self.model.path.trim().is_empty() {
            return Err(AppError::Config("model.path 不能为空".into()));
        }
        if self.source.path.trim().is_empty() {
            return Err(AppError::Config("source.path 不能为空".into()));
        }
        if !matches!(
            self.source.source_type.as_str(),
            "image" | "directory" | "glob" | "video" | "webcam" | "stream"
        ) {
            return Err(AppError::Config(format!(
                "不支持的 source.type: '{}'",
                self.source.source_type
            )));
        }
        Ok(())
    }
}

/// 模型配置。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// ONNX 模型路径；也接受库可识别的模型名（自动下载）。
    pub path: String,
}

/// 输入源配置。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    /// 源类型：image / directory / glob / video / webcam / stream。
    #[serde(rename = "type")]
    pub source_type: String,
    /// 源路径（图片路径 / 目录 / glob / 视频 / 摄像头索引 / 流地址）。
    pub path: String,
    /// 抽帧步长（视频/流专用，1 = 每帧，默认 1）。
    #[serde(default)]
    pub stride: Option<usize>,
    /// 最多处理的帧数（视频/流专用，缺省处理到结尾）。
    #[serde(default)]
    pub max_frames: Option<usize>,
}

/// 推理参数。字段全部可选，未设置时使用库的默认值。
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceSettings {
    /// 硬件设备，如 "cpu"、"cuda:0"、"directml:0"。
    #[serde(default)]
    pub device: Option<String>,
    /// 置信度阈值。
    #[serde(default)]
    pub conf: Option<f32>,
    /// NMS IoU 阈值。
    #[serde(default)]
    pub iou: Option<f32>,
    /// 每图最大检测数。
    #[serde(default)]
    pub max_det: Option<usize>,
    /// 输入尺寸 [高, 宽]。
    #[serde(default)]
    pub imgsz: Option<[usize; 2]>,
    /// 批大小。
    #[serde(default)]
    pub batch: Option<usize>,
    /// 线程数（0 = 全部核心）。
    #[serde(default)]
    pub threads: Option<usize>,
    /// 推理精度，如 "fp16"、"fp32"。
    #[serde(default)]
    pub quantize: Option<String>,
    /// 矩形推理（最小 padding）。
    #[serde(default)]
    pub rect: Option<bool>,
    /// 只保留这些类别 ID。
    #[serde(default)]
    pub classes: Option<Vec<usize>>,
}

impl InferenceSettings {
    /// 映射为库的 `InferenceConfig`。
    pub fn to_inference_config(&self) -> Result<InferenceConfig> {
        let mut cfg = InferenceConfig::default();

        if let Some(conf) = self.conf {
            cfg = cfg.with_confidence(conf);
        }
        if let Some(iou) = self.iou {
            cfg = cfg.with_iou(iou);
        }
        if let Some(max_det) = self.max_det {
            cfg = cfg.with_max_det(max_det);
        }
        if let Some([h, w]) = self.imgsz {
            cfg = cfg.with_imgsz(h, w);
        }
        if let Some(batch) = self.batch {
            cfg = cfg.with_batch(batch);
        }
        if let Some(threads) = self.threads {
            cfg = cfg.with_threads(threads);
        }
        if let Some(q) = &self.quantize {
            let q: Quantization = q
                .parse()
                .map_err(|e| AppError::Config(format!("quantize 非法: {e}")))?;
            cfg = cfg.with_quantize(q);
        }
        if let Some(device) = &self.device {
            let device: Device = device
                .parse()
                .map_err(|e| AppError::Config(format!("device 非法: {e}")))?;
            cfg = cfg.with_device(device);
        }
        if let Some(rect) = self.rect {
            cfg = cfg.with_rect(rect);
        }
        if let Some(classes) = &self.classes {
            cfg = cfg.with_classes(classes.clone());
        }

        Ok(cfg)
    }
}

/// 输出配置。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// 是否保存标注图。
    #[serde(default = "default_true")]
    pub save_annotated: bool,
    /// 是否保存 JSON。
    #[serde(default = "default_true")]
    pub save_json: bool,
    /// 视频源是否逐帧存图（true）而非写回单个标注 mp4（false，默认）。
    #[serde(default = "default_false")]
    pub save_frames: bool,
    /// 输出根目录。
    #[serde(default = "default_output_dir")]
    pub dir: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            save_annotated: true,
            save_json: true,
            save_frames: false,
            dir: default_output_dir(),
        }
    }
}

/// 日志配置。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// 是否打印每张图的详细摘要。
    #[serde(default = "default_true")]
    pub verbose: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self { verbose: true }
    }
}

/// 基准测试配置（`cargo run -- bench config/bench.toml`）。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchConfig {
    /// 模型配置。
    pub model: ModelConfig,
    /// 输入源配置（基准输入集）。
    pub source: SourceConfig,
    /// 推理参数（可选，缺省走库默认值）。
    #[serde(default)]
    pub inference: InferenceSettings,
    /// 基准参数（可选）。
    #[serde(default)]
    pub bench: BenchSettings,
}

impl BenchConfig {
    /// 从文件加载并校验配置。
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| AppError::ConfigIo {
            path: path.display().to_string(),
            source,
        })?;
        let config: Self = toml::from_str(&text)?;
        config.validate()?;
        Ok(config)
    }

    /// 语义校验（TOML 结构之外的约束）。
    fn validate(&self) -> Result<()> {
        if self.model.path.trim().is_empty() {
            return Err(AppError::Config("model.path 不能为空".into()));
        }
        if self.source.path.trim().is_empty() {
            return Err(AppError::Config("source.path 不能为空".into()));
        }
        if self.bench.warmup == 0 || self.bench.iterations == 0 {
            return Err(AppError::Config("bench.warmup / bench.iterations 必须 ≥ 1".into()));
        }
        Ok(())
    }
}

/// 基准参数。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSettings {
    /// 预热遍数（不计时，让后端 / 缓存稳定）。
    #[serde(default = "default_warmup")]
    pub warmup: usize,
    /// 计时遍数。
    #[serde(default = "default_iterations")]
    pub iterations: usize,
    /// 报告输出目录。
    #[serde(default = "default_output_dir")]
    pub dir: String,
}

impl Default for BenchSettings {
    fn default() -> Self {
        Self {
            warmup: default_warmup(),
            iterations: default_iterations(),
            dir: default_output_dir(),
        }
    }
}

/// 精度评估配置（`cargo run -- eval config/eval.toml`）。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalConfig {
    /// 模型配置。
    pub model: ModelConfig,
    /// 数据集（图片目录 + COCO 标注）。
    pub dataset: DatasetConfig,
    /// 推理参数（可选，缺省走库默认值）。
    #[serde(default)]
    pub inference: InferenceSettings,
    /// 评估参数（可选）。
    #[serde(default)]
    pub eval: EvalSettings,
}

impl EvalConfig {
    /// 从文件加载并校验配置。
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| AppError::ConfigIo {
            path: path.display().to_string(),
            source,
        })?;
        let config: Self = toml::from_str(&text)?;
        config.validate()?;
        Ok(config)
    }

    /// 语义校验（TOML 结构之外的约束）。
    fn validate(&self) -> Result<()> {
        if self.model.path.trim().is_empty() {
            return Err(AppError::Config("model.path 不能为空".into()));
        }
        if self.dataset.images.trim().is_empty() {
            return Err(AppError::Config("dataset.images 不能为空".into()));
        }
        if self.dataset.annotations.trim().is_empty() {
            return Err(AppError::Config("dataset.annotations 不能为空".into()));
        }
        Ok(())
    }
}

/// 评估数据集：图片目录 + COCO 标注 JSON。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    /// 图片所在目录（COCO `images[].file_name` 相对于此目录解析）。
    pub images: String,
    /// COCO 标注 JSON 路径。
    pub annotations: String,
}

/// 评估参数。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSettings {
    /// 报告输出目录。
    #[serde(default = "default_output_dir")]
    pub dir: String,
}

impl Default for EvalSettings {
    fn default() -> Self {
        Self {
            dir: default_output_dir(),
        }
    }
}

fn default_warmup() -> usize {
    3
}

fn default_iterations() -> usize {
    20
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_output_dir() -> String {
    "results".to_string()
}
