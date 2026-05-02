#!/usr/bin/env python3
"""Local sanity checks for the PR #1953 frontier lane."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


LANE_DIR = Path(__file__).resolve().parents[1]
UPSTREAM = LANE_DIR / "upstream"
TRAIN = UPSTREAM / "train_gpt.py"
SUBMISSION = UPSTREAM / "submission.json"
CONFIG = LANE_DIR / "configs" / "pr1953.env"
README = LANE_DIR / "README.md"
UPSTREAM_README = UPSTREAM / "README.md"
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


def check_submission_metrics() -> None:
    data = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    if data.get("val_bpb", 9.0) > 1.059:
        fail(f"unexpected val_bpb in submission.json: {data.get('val_bpb')}")
    if data.get("val_bpb_std", 9.0) > 0.0004:
        fail(f"unexpected val_bpb_std in submission.json: {data.get('val_bpb_std')}")
    if data.get("val_loss", 9.0) > 2.317:
        fail(f"unexpected val_loss in submission.json: {data.get('val_loss')}")
    if data.get("seeds") != [42, 0, 1234]:
        fail(f"unexpected seed list: {data.get('seeds')}")

    seed_results = data.get("seed_results", {})
    if sorted(seed_results) != ["0", "1234", "42"]:
        fail(f"unexpected seed_result keys: {sorted(seed_results)}")
    max_artifact = max(row["artifact_bytes"] for row in seed_results.values())
    max_eval_ms = max(row["eval_time_ms"] for row in seed_results.values())
    max_train_ms = max(row["train_wallclock_ms"] for row in seed_results.values())
    if max_artifact >= 16_000_000:
        fail(f"artifact over cap: {max_artifact}")
    if max_eval_ms >= 600_000:
        fail(f"eval over cap: {max_eval_ms}")
    if max_train_ms >= 600_000:
        fail(f"train over cap: {max_train_ms}")


def check_config_values() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    required = {
        "EVAL_SEQ_LEN": "2560",
        "TTT_EVAL_SEQ_LEN": "2560",
        "TTT_MASK": "no_qv",
        "TTT_Q_LORA": "0",
        "TTT_V_LORA": "0",
        "TTT_LOCAL_LR_MULT": "0.75",
        "QK_GAIN_INIT": "5.25",
        "PHASED_TTT_PREFIX_DOCS": "2500",
        "PHASED_TTT_NUM_PHASES": "3",
        "AWQ_LITE_ENABLED": "1",
        "ASYM_LOGIT_RESCALE": "1",
        "GPTQ_RESERVE_SECONDS": "4.0",
        "GPTQ_CALIBRATION_BATCHES": "16",
        "COMPRESSOR": "pergroup",
    }
    for key, value in required.items():
        if f'{key}:-{value}' not in text:
            fail(f"config missing default {key}={value}")


def check_static_code_paths() -> None:
    for needle in [
        "caseops_enabled",
        "val_bytes_files",
        "fineweb_val_bytes_*.bin",
        "softcapped_cross_entropy",
        "SPARSE_ATTN_GATE_ENABLED",
        "SMEAR_GATE_ENABLED",
        "LQER_ENABLED",
        "AWQ_LITE_ENABLED",
        "ASYM_LOGIT_RESCALE",
        "TTT_MASK",
        "QK_GAIN_INIT",
        "eval_val_ttt_phased",
        "PREQUANT_ONLY",
        "TTT_EVAL_ONLY",
    ]:
        require_text(TRAIN, needle)

    train_text = TRAIN.read_text(encoding="utf-8")
    if "NGRAM_TILT_ENABLED" in train_text:
        fail("PR1953 baseline should not already contain n-gram tilt code")
    if train_text.count("not_bos = (input_ids[:, 1:] != BOS_ID)") < 2:
        fail("SmearGate BOS mask is not applied in both forward paths")


def check_log_evidence() -> None:
    expected = {
        "42": "1.05824720",
        "0": "1.05846113",
        "1234": "1.05895276",
    }
    for seed, bpb in expected.items():
        path = UPSTREAM / f"train_seed{seed}.log"
        require_file(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^train_shards: 80$", text, re.MULTILINE):
            fail(f"{path.name} does not show canonical train_shards: 80")
        if not re.search(r"^val_tokens: 47851520$", text, re.MULTILINE):
            fail(f"{path.name} does not show expected canonical val_tokens")
        if f"val_bpb:{bpb}" not in text:
            fail(f"{path.name} missing final bpb {bpb}")


def main() -> None:
    for path in [
        TRAIN,
        SUBMISSION,
        CONFIG,
        README,
        UPSTREAM_README,
        RUNNER,
        BOOTSTRAP,
    ]:
        require_file(path)
    check_submission_metrics()
    check_config_values()
    check_static_code_paths()
    check_log_evidence()
    print("OK: PR1953 lane static checks passed")


if __name__ == "__main__":
    main()
