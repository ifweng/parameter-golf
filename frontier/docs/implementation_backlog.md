# Implementation Backlog

Ordered to maximize signal before cloud spend.

## Immediate

1. Keep `frontier/cuda/train_gpt_frontier.py` as the only readable CUDA source of truth.
2. Finish low-risk same-family ports that do not require a full architecture rewrite:
   - loader improvements from `#1532`
   - schedule knobs from `#1560`
   - env surface for Track B tuning and artifact routing
3. Preserve ancestor reproducibility while making these changes.

## Next

1. Port the `#1530/#1560` family:
   - variable-length attention
   - fused MLP path
   - doc-independent score-first TTT
2. Validate legality boundaries in code comments and notebook entries as features land.
3. Prepare a first cloud replay candidate from the new mainline.

## After the mainline is stable

1. Branch into the `#1518` family:
   - asymmetric two-lane routing
   - wider recurrence over layers `3-5`
   - per-pass embeddings
2. Evaluate whether Tap-In V6 should enter the project or stay as a deferred eval-only path.
3. Revisit `#1552` RecurLoRA only if the recurrence path is stable and under budget.

## Deferred research branches

1. `#1578` tokenizer branch:
   full casefold tokenizer + retokenized dataset + legality review
2. `#1576` GDN-Hybrid branch:
   separate architecture effort, not an incremental extension of the current SP8192 Track B line
