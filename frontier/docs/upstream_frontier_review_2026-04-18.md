# Upstream Frontier Review (2026-04-18)

This updates the active target beyond the April 13 review and folds in the strongest still-open post-April 9 PRs.

## Current Upstream Picture

### Highest-value open PRs

| PR | Claimed result | Seed count | Main gain source | Risk | Decision |
|---|---:|---:|---|---|---|
| `#1698` | `1.00995 BPB` | 3 | aggressive new stack on top of the invalidated `#1687` line | Very high. The PR body reports artifact sizes above the decimal `16,000,000` byte cap and the lineage includes a scoring-bug predecessor. | `defer until independently validated` |
| `#1693` | `1.05733 BPB` | 3 | Casefold V4, AttnOutGate, SmearGate, phased global SGD TTT | High. Strong score, but casefold legality is still pending organizer review. | `watch closely, do not adopt yet` |
| `#1632` | `1.0274 BPB` | 2 | GDN-Hybrid architecture, Track A | Very high. Strong number, but it is an architecture pivot and not the fastest path for the current Track B branch. | `defer until base is stable` |
| `#1585` | `1.0639 BPB` | 3 | casefold tokenizer + parallel residuals + systems optimization | High. Promising transformer-family result, but casefold still dominates the review risk. | `defer until tokenizer branch exists` |
| `#1518` | `1.073938 BPB` | 3 | asymmetric two-lane routing, wider recurrence, per-pass embeddings, Tap-In V6, legal TTT | Medium-high. Rich same-family ideas, but more implementation and review surface. | `test early` |
| `#1530` | `1.07336388 BPB` | 3 | VarLen attention, fused MLP, doc-independent TTT | Medium. Best readable same-family stack to build from. | `must adopt now` |
| `#1560` | `1.07406 BPB` | 3 | tuned `#1530`: warmdown `0.75`, TTT chunk `48`, Muon momentum `0.97` | Medium. Small diff, high practical value. | `must adopt now` |
| `#1532` | `1.0803 BPB` | not restated | async / numpy-first loader, throughput only | Low. Systems-only, already partly ported into the readable frontier script. | `must adopt now` |

### Invalidated or structurally blocked context

| PR | Prior claim | Current status | Practical meaning |
|---|---:|---|---|
| `#1687` | `1.04090 BPB` | closed after a scoring bug | Do not use as a trusted target. Any descendant must be independently re-validated. |
| `#1576` | `1.01671233 BPB` | closed | Strong historical result, but not an open same-family target. |
| `#1578` | `1.06681663 BPB` | open, but casefold-dependent | Useful as tokenizer evidence, not as the current lowest-friction implementation path. |

## Ranked target-stack decision

### `must adopt now`

1. Use `#1530/#1560` as the primary readable Track B implementation target.
2. Keep `#1532`-style throughput work in the port queue because it is low-risk and orthogonal.
3. Treat the April 5 readable SP8192 script only as a starting surface, not as the target architecture.

### `test early`

1. Wider recurrence and per-pass embeddings from `#1518`.
2. The safer schedule and systems deltas that can be isolated from `#1560` and `#1532`.

### `defer until base is stable`

1. Any casefold tokenizer path: `#1693`, `#1585`, `#1578`.
2. GDN-Hybrid / Track A path from `#1632`.
3. Any lineage that depends on the invalidated `#1687` scoring path until independently reproduced.

### `ignore for the current branch`

1. Non-frontier SP1024-only changes that do not transfer to the Track B target stack.
2. PRs without credible multi-seed evidence or with no clear readable path to reproduction.

## Updated project target

- **Best open same-family implementation target:** `#1530/#1560`
- **Best open low-risk throughput delta:** `#1532`
- **Highest-upside same-family extension after the base is stable:** `#1518`
- **Watchlist only:** `#1693`, `#1698`, `#1632`, `#1585`

Practical project stance:
- keep the current branch on readable Track B transformer-family work
- do not chase casefold or new-architecture results before we have a stable same-family reproduction path
- use local MLX only for cheap directionality; reserve real score claims for CUDA/H100
