#!/usr/bin/env python3
"""Local sanity checks for the PR #1855 frontier lane.

These checks are intentionally lightweight. They do not prove compliance and
they do not run CUDA training. They catch bad provenance, missing files, broken
CaseOps roundtrip behavior, and obvious BPB/accounting path regressions before
we spend H100 time.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path


LANE_DIR = Path(__file__).resolve().parents[1]
UPSTREAM = LANE_DIR / "upstream"
TRAIN = UPSTREAM / "train_gpt.py"
PREP = UPSTREAM / "prepare_caseops_data.py"
LOSSLESS = UPSTREAM / "lossless_caps.py"
TOKENIZER = (
    UPSTREAM
    / "tokenizers"
    / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
)
SUBMISSION = UPSTREAM / "submission.json"
CONFIG = LANE_DIR / "configs" / "pr1855.env"
README = LANE_DIR / "README.md"


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"WARN: {message}")


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"missing required file: {path}")


def require_text(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        fail(f"{path.name} missing required text: {needle}")


def import_lossless_caps():
    spec = importlib.util.spec_from_file_location("pr1855_lossless_caps", LOSSLESS)
    if spec is None or spec.loader is None:
        fail(f"cannot import {LOSSLESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_submission_metrics() -> None:
    data = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    if data.get("val_bpb", 9.0) > 1.062:
        fail(f"unexpected val_bpb in submission.json: {data.get('val_bpb')}")
    if data.get("artifact_bytes_max", 99_999_999) >= 16_000_000:
        fail(f"artifact over cap in submission.json: {data.get('artifact_bytes_max')}")
    if data.get("train_steps_mean", 0) < 4900:
        fail(f"unexpected train_steps_mean: {data.get('train_steps_mean')}")
    seeds = data.get("seeds")
    if seeds != [42, 0, 1234]:
        fail(f"unexpected seed list: {seeds}")
    seed_results = data.get("seed_results", {})
    eval_times = [row.get("eval_time_s", 9999) for row in seed_results.values()]
    if not eval_times or max(eval_times) >= 600:
        fail(f"seed eval_time_s over cap: {eval_times}")


def check_caseops_roundtrip() -> None:
    if sys.version_info < (3, 10):
        warn("Python <3.10, skipping CaseOps roundtrip that uses zip(strict=True)")
        return
    module = import_lossless_caps()
    enc = getattr(module, "encode_lossless_caps_v2")
    dec = getattr(module, "decode_lossless_caps_v2")
    samples = [
        "Hello WORLD. NASA launched A/B tests.",
        "FineWeb has MIXED Case, codeLikeIdentifiers, and UTF-8 cafe.",
        "already lowercase plus Title Case and ALLCAPS",
    ]
    for sample in samples:
        encoded = enc(sample)
        decoded = dec(encoded)
        if decoded != sample:
            fail(f"CaseOps roundtrip mismatch: {sample!r} -> {decoded!r}")


def check_tokenizer_if_available() -> None:
    try:
        import sentencepiece as spm
    except Exception as exc:  # pragma: no cover - optional local dependency
        warn(f"sentencepiece unavailable, skipping tokenizer load: {exc}")
        return
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    if int(sp.vocab_size()) != 8192:
        fail(f"tokenizer vocab mismatch: {sp.vocab_size()}")


def check_config_values() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    required = {
        "COMPRESSOR": "pergroup",
        "PHASED_TTT_PREFIX_DOCS": "2500",
        "PHASED_TTT_NUM_PHASES": "3",
        "MLP_CLIP_SIGMAS": "11.5",
        "EMBED_CLIP_SIGMAS": "14.0",
        "WARMDOWN_FRAC": "0.85",
        "BETA2": "0.99",
        "TTT_BETA2": "0.99",
        "TTT_WEIGHT_DECAY": "0.5",
        "TTT_LORA_RANK": "80",
        "SPARSE_ATTN_GATE_SCALE": "0.5",
        "SMEAR_GATE_ENABLED": "1",
        "LQER_ENABLED": "1",
        "LQER_TOP_K": "3",
    }
    for key, value in required.items():
        if f'{key}:-{value}' not in text:
            fail(f"config missing default {key}={value}")


def check_static_code_paths() -> None:
    require_text(TRAIN, "caseops_enabled")
    require_text(TRAIN, "val_bytes_files")
    require_text(TRAIN, "fineweb_val_bytes_*.bin")
    require_text(TRAIN, "softcapped_cross_entropy")
    require_text(TRAIN, "SPARSE_ATTN_GATE_ENABLED")
    require_text(TRAIN, "SMEAR_GATE_ENABLED")
    require_text(TRAIN, "LQER_ENABLED")
    require_text(TRAIN, "COMPRESSOR")
    require_text(TRAIN, "pergroup")
    require_text(TRAIN, "lrzip")
    require_text(TRAIN, "eval_val_ttt_phased")

    train_text = TRAIN.read_text(encoding="utf-8")
    if train_text.count("not_bos = (input_ids[:, 1:] != BOS_ID)") < 2:
        fail("SmearGate BOS mask is not applied in both forward paths")

    prep_text = PREP.read_text(encoding="utf-8")
    if not re.search(r"^BOS_ID\s*=\s*1\b", prep_text, re.MULTILINE):
        fail("prepare_caseops_data.py does not define BOS_ID = 1")
    if "val_buf_bytes.append(0)" not in prep_text:
        fail("prepare_caseops_data.py does not set BOS byte count to 0")


def main() -> None:
    for path in [TRAIN, PREP, LOSSLESS, TOKENIZER, SUBMISSION, CONFIG, README]:
        require_file(path)
    check_submission_metrics()
    check_caseops_roundtrip()
    check_tokenizer_if_available()
    check_config_values()
    check_static_code_paths()
    print("OK: PR1855 lane static checks passed")


if __name__ == "__main__":
    main()
