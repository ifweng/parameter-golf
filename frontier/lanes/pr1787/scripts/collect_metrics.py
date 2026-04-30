#!/usr/bin/env python3
"""Parse frontier cloud logs into compact JSON and Markdown summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "train_stop_ms": re.compile(
        r"stopping_early: wallclock_cap train_time: (?P<value>[0-9]+)ms"
    ),
    "steps": re.compile(r"stopping_early: .* step: (?P<value>[0-9]+)/"),
    "pre_quant": re.compile(
        r"diagnostic pre-quantization post-ema "
        r"val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+) "
        r"eval_time:(?P<ms>[0-9]+)ms"
    ),
    "quantized": re.compile(
        r"diagnostic quantized "
        r"val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+) "
        r"eval_time:(?P<ms>[0-9]+)ms"
    ),
    "post_ttt": re.compile(
        r"quantized_ttt_phased "
        r"val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+) "
        r"eval_time:(?P<ms>[0-9]+)ms"
    ),
    "artifact_bytes": re.compile(
        r"Total submission size quantized\+(?:brotli|pergroup|lzma): (?P<value>[0-9]+) bytes"
    ),
    "code_bytes": re.compile(r"Code size \(compressed\): (?P<value>[0-9]+) bytes"),
    "val_tokens": re.compile(r"^val_tokens: (?P<value>[0-9]+)$", re.MULTILINE),
    "target_tokens": re.compile(r"\btarget_tokens:(?P<value>[0-9]+)\b"),
    "gptq_reserve": re.compile(
        r"gptq:reserving (?P<reserve>[0-9.]+)s, effective=(?P<effective>[0-9.]+)ms"
    ),
}


def _last_match(pattern: re.Pattern[str], text: str):
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def parse_log(path: Path, run_root: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    seed_match = re.search(r"train_seed(?P<seed>[0-9]+)\.log$", path.name)
    seed = int(seed_match.group("seed")) if seed_match else None
    row: dict[str, object] = {"seed": seed, "log": str(path)}

    for key in [
        "train_stop_ms",
        "steps",
        "artifact_bytes",
        "code_bytes",
        "val_tokens",
        "target_tokens",
    ]:
        match = _last_match(PATTERNS[key], text)
        if match:
            row[key] = int(match.group("value"))

    match = _last_match(PATTERNS["gptq_reserve"], text)
    if match:
        row["gptq_reserve_s"] = float(match.group("reserve"))
        row["gptq_effective_train_ms"] = float(match.group("effective"))

    for key in ["pre_quant", "quantized", "post_ttt"]:
        match = _last_match(PATTERNS[key], text)
        if match:
            row[f"{key}_val_loss"] = float(match.group("loss"))
            row[f"{key}_val_bpb"] = float(match.group("bpb"))
            row[f"{key}_eval_ms"] = int(match.group("ms"))

    # The upstream PR #1787 evidence stores the final PR #1767 TTT-only eval in
    # JSON files. Prefer those when present. Direct cloud runs will usually only
    # have the end-to-end train_seed*.log values.
    if seed is not None:
        result_path = run_root / "ttt_pr1767" / f"seed_{seed}_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row["post_ttt_val_loss"] = float(result["post_ttt_val_loss"])
            row["post_ttt_val_bpb"] = float(result["post_ttt_val_bpb"])
            row["post_ttt_eval_ms"] = int(result["post_ttt_eval_time_ms"])
            row["post_ttt_source"] = str(result_path)

    if "post_ttt_val_bpb" in row:
        row["status"] = "complete"
    elif "quantized_val_bpb" in row:
        row["status"] = "quantized_only"
    elif "pre_quant_val_bpb" in row:
        row["status"] = "prequant_only"
    else:
        row["status"] = "incomplete"
    return row


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def write_markdown_table(
    path: Path, run_name: str, rows: list[dict[str, object]], headers: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    md_headers = ["run", *headers]
    lines = [
        "# Metrics Summary",
        "",
        f"Run root: `{run_name}`",
        "",
        "| " + " | ".join(md_headers) + " |",
        "| " + " | ".join("---" for _ in md_headers) + " |",
    ]
    for row in rows:
        cells = [run_name, *(_format_cell(row.get(header)) for header in headers)]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional output path. Defaults to RUN_ROOT/metrics_summary.json.",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=None,
        help="Optional Markdown output path. Defaults to RUN_ROOT/metrics_summary.md.",
    )
    args = parser.parse_args()

    logs = sorted(args.run_root.glob("train_seed*.log"))
    if not logs:
        raise SystemExit(f"no train_seed*.log files found under {args.run_root}")

    rows = [parse_log(path, args.run_root) for path in logs]
    summary_path = args.summary_path or args.run_root / "metrics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.markdown_path or args.run_root / "metrics_summary.md"

    headers = [
        "seed",
        "status",
        "steps",
        "pre_quant_val_loss",
        "pre_quant_val_bpb",
        "quantized_val_loss",
        "quantized_val_bpb",
        "post_ttt_val_loss",
        "post_ttt_val_bpb",
        "val_tokens",
        "target_tokens",
        "artifact_bytes",
        "code_bytes",
        "train_stop_ms",
        "post_ttt_eval_ms",
        "gptq_reserve_s",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row.get(header, "")) for header in headers))
    write_markdown_table(markdown_path, args.run_root.name, rows, headers)
    print(f"wrote {summary_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
