#!/usr/bin/env python3
"""Static checks for PR #2014 exp01: slope 0.3 + reverse-Cholesky GPTQ."""

from __future__ import annotations

import sys
from pathlib import Path


LANE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LANE_DIR.parents[2]
BASE_LANE = REPO_ROOT / "frontier" / "lanes" / "pr2014"
TRAIN = LANE_DIR / "train_gpt.py"
CONFIG = LANE_DIR / "configs" / "exp01.env"
RUNNER = LANE_DIR / "run_8xh100.sh"
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


def check_patch_surface() -> None:
    text = TRAIN.read_text(encoding="utf-8")
    required = [
        'LEAKY_RELU_SQ_SLOPE = float(os.environ.get("LEAKY_RELU_SQ_SLOPE", "0.3"))',
        "LEAKY_RELU_SQ_NEG_DERIV_SCALE = 2.0 * LEAKY_RELU_SQ_SLOPE * LEAKY_RELU_SQ_SLOPE",
        'GPTQ_HINV_MODE = os.environ.get("GPTQ_HINV_MODE", "reverse_cholesky").strip().lower()',
        "SLOPE=LEAKY_RELU_SQ_SLOPE",
        "NEG_DERIV_SCALE=LEAKY_RELU_SQ_NEG_DERIV_SCALE",
        "negative_slope=LEAKY_RELU_SQ_SLOPE",
        'if GPTQ_HINV_MODE == "reverse_cholesky":',
        "torch.linalg.solve_triangular",
        "gptq_hinv_mode",
        "leaky_relu_sq_slope",
    ]
    for needle in required:
        if needle not in text:
            fail(f"train_gpt.py missing exp01 patch marker: {needle}")
    if "negative_slope=0.5" in text:
        fail("train_gpt.py still has hard-coded eager negative_slope=0.5")


def check_config() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    for needle in [
        "../../pr2014/configs/pr2014.env",
        'LEAKY_RELU_SQ_SLOPE="${LEAKY_RELU_SQ_SLOPE:-0.3}"',
        'GPTQ_HINV_MODE="${GPTQ_HINV_MODE:-reverse_cholesky}"',
    ]:
        if needle not in text:
            fail(f"exp01 config missing {needle}")


def check_reverse_cholesky_equivalence() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional local dependency
        warn(f"torch unavailable, skipping reverse-Cholesky equivalence check: {exc}")
        return
    torch.manual_seed(123)
    n = 32
    a = torch.randn(n, n, dtype=torch.float32)
    h = a @ a.T + 0.1 * torch.eye(n, dtype=torch.float32)
    baseline = torch.cholesky_inverse(torch.linalg.cholesky(h))
    baseline = torch.linalg.cholesky(baseline, upper=True)
    h_flip = torch.flip(h, dims=(0, 1))
    l_flip = torch.linalg.cholesky(h_flip)
    u = torch.flip(l_flip, dims=(0, 1)).contiguous()
    candidate = torch.linalg.solve_triangular(
        u, torch.eye(n, dtype=torch.float32), upper=True
    )
    if not torch.allclose(candidate, baseline, rtol=2e-4, atol=2e-5):
        diff = (candidate - baseline).abs().max().item()
        fail(f"reverse-Cholesky equivalence check failed; max_abs_diff={diff}")


def main() -> None:
    for path in [
        TRAIN,
        CONFIG,
        RUNNER,
        README,
        BASE_LANE / "upstream" / "submission.json",
        BASE_LANE / "upstream" / "tokenizers" / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
    ]:
        require_file(path)
    require_text(RUNNER, "PR2014 exp01 lane")
    check_patch_surface()
    check_config()
    check_reverse_cholesky_equivalence()
    print("OK: PR2014 exp01 lane static checks passed")


if __name__ == "__main__":
    main()
