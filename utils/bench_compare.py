#!/usr/bin/env python3
"""汇总多个 bench_report.json，打印跨设备 / 跨 batch / 跨量化的 Markdown 对比表。

用法（在项目根目录）：
    python utils/bench_compare.py [results/bench_cpu results/bench_cuda ...]

缺省扫描 `results/bench_*/bench_report.json`。输出按吞吐降序排列，
便于直观对比 CPU vs CUDA vs TensorRT、batch=1/4/8、fp16/fp32。
"""
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else \
        sorted(Path("results").glob("bench_*/bench_report.json"))
    if not paths:
        print("没有找到 bench_report.json（先跑 `cargo run -- bench config/bench_*.toml`）")
        return 1

    rows = []
    for p in paths:
        r = load(p)
        lat = r["latency_ms"]["total"]
        rows.append({
            "报告": p.parent.name,
            "设备": r["device"],
            "batch": r["batch_size"],
            "p50(ms)": round(lat["p50"], 1),
            "p95(ms)": round(lat["p95"], 1),
            "吞吐(图/s)": round(r["throughput"]["images_per_sec"], 2),
            "峰值RSS(MiB)": round(r["memory"]["peak_rss_mib"], 1),
        })

    rows.sort(key=lambda x: -x["吞吐(图/s)"])

    print("| 报告 | 设备 | batch | p50(ms) | p95(ms) | 吞吐(图/s) | 峰值RSS(MiB) |")
    print("|------|------|-------|---------|---------|------------|--------------|")
    for x in rows:
        print(f"| {x['报告']} | {x['设备']} | {x['batch']} | "
              f"{x['p50(ms)']} | {x['p95(ms)']} | {x['吞吐(图/s)']} | {x['峰值RSS(MiB)']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
