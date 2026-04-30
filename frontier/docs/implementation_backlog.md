# Implementation Backlog

Ordered to maximize signal before cloud spend.

## Immediate

1. Keep `frontier/lanes/pr2014/upstream/train_gpt.py` frozen as the exact open-frontier reproduction source.
2. Use `frontier/lanes/pr2014/configs/pr2014.env` for reproducible cloud runs and config-only smoke tests.
3. Run the PR #2014 static checker and cheap 1-GPU smoke tests before any full 8xH100 run.
4. Preserve `frontier/lanes/pr1855` as the accepted leaderboard control.

## Next

1. Create config overlays or separate lanes for low-byte #2014 deltas:
   - PR #1948 reverse-Cholesky GPTQ
   - PR #1948 LeakyReLU-square slope `0.3`
   - stricter GPTQ reserve values, likely `8s` or `12s`
   - short-doc TTT schedule variants
   - progressive 3k context schedule variants
2. Validate legality boundaries in code comments, metrics summaries, and the experiment ledger.
3. Promote only one to three candidates to full 8xH100 confirmation.

## After the mainline is stable

1. Revisit legal eval-time adaptation variants if they are still score-first and single-pass.
2. Explore compression-aware or quantization-aware training only after the #2014 reproduction is stable.
3. Consider architecture pivots only as non-leaderboard research branches.

## Deferred research branches

1. PR #1967 n-gram tilt:
   defer unless all precompute and scoring are timer-inclusive and accepted under issue `#1017`
2. Mamba/SSM or JEPA/text-diffusion branches:
   useful for a noticeable research implementation, not the first leaderboard push
3. Alternative reversible tokenization beyond CaseOps:
   high upside but high compliance burden
