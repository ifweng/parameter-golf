#!/usr/bin/env python3
"""Local sanity checks for the PR #2014 frontier lane.

These checks are intentionally lightweight. They do not run CUDA training.
They verify provenance, published metrics, CaseOps reversibility, full
validation-target accounting, and the #2014 config knobs before cloud spend.
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
CONFIG = LANE_DIR / "configs" / "pr2014.env"
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
    spec = importlib.util.spec_from_file_location("pr2014_lossless_caps", LOSSLESS)
    if spec is None or spec.loader is None:
        fail(f"cannot import {LOSSLESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_submission_metrics() -> None:
    data = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    if data.get("val_bpb", 9.0) > 1.058:
        fail(f"unexpected val_bpb in submission.json: {data.get('val_bpb')}")
    if data.get("artifact_bytes_max", 99_999_999) >= 16_000_000:
        fail(f"artifact over cap in submission.json: {data.get('artifact_bytes_max')}")
    if data.get("train_wallclock_ms_max", 999_999) >= 600_000:
        fail(f"train wallclock over cap: {data.get('train_wallclock_ms_max')}")
    if data.get("eval_time_s_max", 9999) >= 600:
        fail(f"eval_time_s over cap: {data.get('eval_time_s_max')}")
    seeds = data.get("seeds")
    if seeds != [42, 314, 0]:
        fail(f"unexpected seed list: {seeds}")
    seed_results = data.get("seed_results", {})
    for seed in ["42", "314", "0"]:
        row = seed_results.get(seed)
        if not isinstance(row, dict):
            fail(f"missing seed result: {seed}")
        if row.get("val_tokens") != 47_853_343 or row.get("target_tokens") != 47_853_343:
            fail(f"seed {seed} does not cover full validation targets: {row}")


def check_upstream_logs() -> None:
    logs = sorted(UPSTREAM.glob("train_seed*.log"))
    if len(logs) != 3:
        fail(f"expected 3 upstream logs, found {len(logs)}")
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        val_tokens = re.findall(r"^val_tokens: ([0-9]+)$", text, re.MULTILINE)
        target_tokens = re.findall(r"\btarget_tokens:([0-9]+)\b", text)
        if not val_tokens or int(val_tokens[-1]) != 47_853_343:
            fail(f"{path.name} missing full val_tokens coverage")
        if not target_tokens or int(target_tokens[-1]) != 47_853_343:
            fail(f"{path.name} missing full target_tokens coverage")
        if "quantized_ttt_phased val_loss:" not in text:
            fail(f"{path.name} missing post-TTT metric")


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
        "TRAIN_SEQ_LEN": "3072",
        "ROPE_TRAIN_SEQ_LEN": "3072",
        "TRAIN_SEQ_SCHEDULE": "1024@0.100,2048@0.700,3072@1.000",
        "EVAL_INCLUDE_TAIL": "1",
        "EVAL_SEQ_LEN": "3072",
        "EVAL_STRIDE": "1536",
        "TTT_EVAL_SEQ_LEN": "3072",
        "TTT_SHORT_SCORE_FIRST_ENABLED": "1",
        "TTT_SHORT_SCORE_FIRST_STEPS": "256:8,2000:24",
        "TTT_MASK": "no_qv",
        "TTT_Q_LORA": "0",
        "TTT_V_LORA": "0",
        "TTT_LOCAL_LR_MULT": "0.75",
        "PHASED_TTT_PREFIX_DOCS": "2500",
        "PHASED_TTT_NUM_PHASES": "1",
        "QK_GAIN_INIT": "5.25",
        "GPTQ_RESERVE_SECONDS": "4.0",
        "AWQ_LITE_ENABLED": "1",
        "ASYM_LOGIT_RESCALE": "1",
        "LQER_ENABLED": "1",
    }
    for key, value in required.items():
        if f'{key}:-{value}' not in text:
            fail(f"config missing default {key}={value}")


def check_static_code_paths() -> None:
    for needle in [
        "caseops_enabled",
        "val_bytes_files",
        "fineweb_val_bytes_*.bin",
        "TRAIN_SEQ_SCHEDULE",
        "EVAL_INCLUDE_TAIL",
        "TTT_SHORT_SCORE_FIRST_STEPS",
        "TTT_MASK",
        "QK_GAIN_INIT",
        "AWQ_LITE_ENABLED",
        "ASYM_LOGIT_RESCALE",
        "GPTQ_RESERVE_SECONDS",
        "target_tokens",
        "eval_val_ttt_phased",
        "pergroup",
        "lrzip",
    ]:
        require_text(TRAIN, needle)

    train_text = TRAIN.read_text(encoding="utf-8")
    if train_text.count("not_bos = (input_ids[:, 1:] != BOS_ID)") < 2:
        fail("SmearGate BOS mask is not applied in both forward paths")
    if "torch.cholesky_inverse(torch.linalg.cholesky(H))" in train_text:
        warn("baseline still uses the older GPTQ Cholesky path; reverse-Cholesky is an improvement candidate")
    if "negative_slope=0.5" in train_text:
        warn("baseline still uses LeakyReLU-square slope 0.5; PR #1948 slope 0.3 is an improvement candidate")

    prep_text = PREP.read_text(encoding="utf-8")
    if not re.search(r"^BOS_ID\s*=\s*1\b", prep_text, re.MULTILINE):
        fail("prepare_caseops_data.py does not define BOS_ID = 1")
    if "val_buf_bytes.append(0)" not in prep_text:
        fail("prepare_caseops_data.py does not set BOS byte count to 0")


def main() -> None:
    for path in [TRAIN, PREP, LOSSLESS, TOKENIZER, SUBMISSION, CONFIG, README]:
        require_file(path)
    check_submission_metrics()
    check_upstream_logs()
    check_caseops_roundtrip()
    check_tokenizer_if_available()
    check_config_values()
    check_static_code_paths()
    print("OK: PR2014 lane static checks passed")


if __name__ == "__main__":
    main()
