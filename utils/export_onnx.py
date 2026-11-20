#!/usr/bin/env python
"""批量把 Ultralytics .pt 权重导出为 ONNX（覆盖同名 .onnx）。

用法:
    python utils/export_onnx.py models/seg/yolo26n-seg.pt models/sem/yolo26n-sem.pt ...

依赖: 已安装 ultralytics 的 Python 环境（本项目用 yolo_algorithm_26）。
"""
import sys
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    failed = 0
    for raw in sys.argv[1:]:
        src = Path(raw)
        print(f"\n=== 导出 {src} -> {src.with_suffix('.onnx')} ===")
        try:
            YOLO(str(src)).export(format="onnx")
        except Exception as exc:  # noqa: BLE001 - 打印并继续导出下一个
            failed += 1
            print(f"  导出失败: {exc}")
        else:
            print(f"  完成: {src.with_suffix('.onnx')}")

    print(f"\n全部完成，失败 {failed} 个")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
