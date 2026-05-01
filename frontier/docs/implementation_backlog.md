# Implementation Backlog

Ordered to maximize signal before cloud spend.

## Immediate

1. Keep `frontier/lanes/pr1855` frozen as the accepted leaderboard control.
2. Use `frontier/lanes/pr1855_exp01_awqlite_asymlogit` as the main improvement lane.
3. Reproduce the PR #2101-style PR1855 improvement before any original deltas.
4. Treat PR #2014 and `pr2014_exp01_slope03_rchgptq` as stopped unless needed for forensic comparison.

## Next

1. Run the PR1855 Exp01 static checker and a cheap smoke:
   - `python3 frontier/lanes/pr1855_exp01_awqlite_asymlogit/checks/check_pr1855_exp01_lane.py`
   - `PREQUANT_ONLY=1`, `MAX_WALLCLOCK_SECONDS=120`, `COMPRESSOR=brotli`
2. Run one full seed 42 on 8xH100 and compare post-TTT BPB, artifact bytes, and eval time against reproduced PR1855.
3. If seed 42 is healthy, run three seeds: `42 0 1234`.
4. Only after reproduction, test the PR #2060 retune as a config-only A/B:
   - `MATRIX_LR=0.028`
   - `LQER_RANK=2`
   - `LQER_ASYM_GROUP=32`
   - `LQER_TOP_K=4`
   - `TTT_LOCAL_LR_MULT=0.80`
5. Validate legality boundaries in code comments, metrics summaries, and the experiment ledger.

## After the mainline is stable

1. Revisit legal eval-time adaptation variants if they are still score-first and single-pass.
2. Explore compression-aware or quantization-aware training only after the PR1855 Exp01 reproduction is stable.
3. Consider architecture pivots only as non-leaderboard research branches.

## Deferred research branches

1. PR #2018 / PR #1967 n-gram tilt:
   defer because the user does not want it as the base; only reopen as a separate high-risk lane
2. Mamba/SSM or JEPA/text-diffusion branches:
   useful for a noticeable research implementation, not the first leaderboard push
3. Alternative reversible tokenization beyond CaseOps:
   high upside but high compliance burden
