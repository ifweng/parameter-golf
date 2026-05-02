# Implementation Backlog

Ordered to maximize signal before cloud spend.

## Immediate

1. Keep `frontier/lanes/pr1855` frozen as the accepted leaderboard control.
2. Use `frontier/lanes/pr1953` as the active clean baseline.
3. Treat the previous PR1855 Exp03 loss-gated TTT result as negative evidence because the observed average score was `1.06076055`, worse than PR #1953 and the intended PR1855-derived target.
4. Keep `frontier/lanes/pr1855_exp01_awqlite_asymlogit` for forensic comparison only, not as the default route.
5. Treat PR #2014 and `pr2014_exp01_slope03_rchgptq` as stopped unless needed for forensic comparison.

## Next

1. Run the PR1953 static checker:
   - `python3 frontier/lanes/pr1953/checks/check_pr1953_lane.py`
2. Run a cheap PR1953 smoke:
   - `PREQUANT_ONLY=1`, `MAX_WALLCLOCK_SECONDS=120`, `COMPRESSOR=brotli`
3. Run one full PR1953 seed 42 on 8xH100 and compare post-TTT BPB, artifact bytes, and eval time against upstream seed 42.
4. If seed 42 reproduces directionally, run the three PR1953 seeds: `42 0 1234`.
5. Test `frontier/lanes/pr1953_exp01_weighted_ttt` first, because the current observed gap is mostly TTT recovery rather than pre-quant or quantized base quality.
6. If a PR1953 artifact exists, run weighted TTT as `TTT_EVAL_ONLY=1` before spending on a full retrain.
7. Create `frontier/lanes/pr1953_exp01_entropy_tilt` only after the weighted-TTT result is negative or after it establishes a stronger local-TTT baseline.
8. Validate legality boundaries in code comments, metrics summaries, and the experiment ledger before any three-seed novel run.

## After the mainline is stable

1. Revisit legal eval-time adaptation variants if they are still score-first and single-pass.
2. Explore compression-aware or quantization-aware training only after the PR1855 Exp01 reproduction is stable.
3. Consider architecture pivots only as non-leaderboard research branches.

## Deferred research branches

1. Fixed-boost PR #2018 / PR #1967 n-gram tilt:
   use only as a control for the PR1953 entropy-adaptive token-only tilt lane
2. Mamba/SSM or JEPA/text-diffusion branches:
   useful for a noticeable research implementation, not the first leaderboard push
3. Alternative reversible tokenization beyond CaseOps:
   high upside but high compliance burden
