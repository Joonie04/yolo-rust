//! 构建脚本：`video` feature 启用时，把 FFmpeg 运行时 DLL 拷到 exe 旁边。
//!
//! 底层 ort 用 `copy-dylibs` 把 ONNX Runtime 的 DLL 拷到 target/<profile>/，
//! 这里对 FFmpeg 做同样的事，使产物自带运行库、不依赖系统 PATH 或外部环境。
//! DLL 来源为 `FFMPEG_DIR/bin`（由 `.cargo/config.toml` 指向 `third_party/ffmpeg`）。

fn main() {
    #[cfg(feature = "video")]
    copy_ffmpeg_dlls();
}

#[cfg(feature = "video")]
fn copy_ffmpeg_dlls() {
    let Ok(ffmpeg_dir) = std::env::var("FFMPEG_DIR") else {
        return;
    };
    let Ok(out_dir) = std::env::var("OUT_DIR") else {
        return;
    };

    // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out，三级上级即 target/<profile>/。
    let profile_dir = std::path::Path::new(&out_dir).join("../../..");
    let bin = std::path::Path::new(&ffmpeg_dir).join("bin");

    let Ok(entries) = std::fs::read_dir(&bin) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_dll = path
            .extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("dll"));
        if !is_dll {
            continue;
        }
        if let Some(name) = path.file_name() {
            let _ = std::fs::copy(&path, profile_dir.join(name));
        }
    }

    // FFmpeg 目录或 DLL 变化时重新执行本脚本。
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    println!("cargo:rerun-if-changed={}", bin.display());
}
