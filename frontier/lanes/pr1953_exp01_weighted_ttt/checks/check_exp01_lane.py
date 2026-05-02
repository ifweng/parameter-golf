#!/usr/bin/env python3
"""Local sanity checks for PR #1953 Exp01 weighted TTT."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


LANE_DIR = Path(__file__).resolve().parents[1]
BASE_LANE = LANE_DIR.parent / "pr1953"
TRAIN = LANE_DIR / "upstream" / "train_gpt.py"
SUBMISSION = LANE_DIR / "upstream" / "submission.json"
CONFIG = LANE_DIR / "configs" / "exp01_weighted_ttt.env"
README = LANE_DIR / "README.md"
RUNNER = LANE_DIR / "run_8xh100.sh"
BOOTSTRAP = LANE_DIR / "bootstrap_workspace_data.sh"


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"missing required file: {path}")


def require_text(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        fail(f"{path.name} missing required text: {needle}")


def check_base_metrics() -> None:
    data = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    if data.get("val_bpb", 9.0) > 1.059:
        fail(f"unexpected inherited val_bpb: {data.get('val_bpb')}")
    if data.get("seeds") != [42, 0, 1234]:
        fail(f"unexpected inherited seed list: {data.get('seeds')}")


def check_config() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    for needle in [
        "frontier/lanes/pr1953/configs/pr1953.env",
        'TTT_LOSS_WEIGHTED_ENABLED="${TTT_LOSS_WEIGHTED_ENABLED:-1}"',
        'TTT_LOSS_WEIGHT_ALPHA="${TTT_LOSS_WEIGHT_ALPHA:-0.5}"',
        'TTT_LOSS_WEIGHT_MIN="${TTT_LOSS_WEIGHT_MIN:-0.5}"',
        'TTT_LOSS_WEIGHT_MAX="${TTT_LOSS_WEIGHT_MAX:-1.75}"',
        'TTT_LOSS_WEIGHT_EMA="${TTT_LOSS_WEIGHT_EMA:-0.98}"',
    ]:
        if needle not in text:
            fail(f"config missing {needle}")


def check_weighted_ttt_code() -> None:
    for needle in [
        "TTT_LOSS_WEIGHTED_ENABLED",
        "TTT_LOSS_WEIGHT_ALPHA",
        "TTT_LOSS_WEIGHT_MIN",
        "TTT_LOSS_WEIGHT_MAX",
        "TTT_LOSS_WEIGHT_EMA",
        "def _ttt_loss_update_weights",
        "ttt_loss_weighted:",
        "loss_update_weights",
        "per_doc * loss_update_weights",
        "lw:",
    ]:
        require_text(TRAIN, needle)

    text = TRAIN.read_text(encoding="utf-8")
    score_pos = text.find("_accumulate_bpb(")
    train_pos = text.find("per_doc * loss_update_weights")
    if score_pos < 0 or train_pos < 0 or score_pos > train_pos:
        fail("weighted TTT update does not occur after score accumulation")
    if "NGRAM_TILT_ENABLED" in text:
        fail("Exp01 should not add n-gram tilt yet")


def check_log_evidence() -> None:
    for seed in ["42", "0", "1234"]:
        path = LANE_DIR / "upstream" / f"train_seed{seed}.log"
        require_file(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^train_shards: 80$", text, re.MULTILINE):
            fail(f"{path.name} does not show canonical train_shards: 80")


def main() -> None:
    for path in [BASE_LANE / "configs" / "pr1953.env", TRAIN, SUBMISSION, CONFIG, README, RUNNER, BOOTSTRAP]:
        require_file(path)
    check_base_metrics()
    check_config()
    check_weighted_ttt_code()
    check_log_evidence()
    print("OK: PR1953 Exp01 weighted TTT static checks passed")


if __name__ == "__main__":
    main()
