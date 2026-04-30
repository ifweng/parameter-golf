# Upstream Frontier Review (2026-04-13)

This review updates the project target beyond the local merged README.

## Current Upstream Picture

### Highest-value pending PRs

| PR | Claimed result | Seed count | Main gain source | Readable / reproducible | Risk | Decision |
|---|---:|---:|---|---|---|---|
| `#1560` | `1.07406 BPB` | 3 | VarLen attention, fused MLP, doc-TTT, warmdown / chunk tuning on top of `#1530` | Medium. Large diff but concrete and measurable. | Medium | `must adopt now` as the primary Track B target family |
| `#1518` | `1.073938 BPB` | 3 | asymmetric two-lane routing, wider recurrence, per-pass embeddings, Tap-In V6, TTT | Medium-low. Detailed writeup but more moving parts and custom eval helpers. | Medium-high | `test early` |
| `#1532` | `1.0803 BPB` | not fully restated, based on `#1493` | asynchronous / numpy-first loader, throughput only | High for the loader idea itself; low modeling ambiguity | Low | `must adopt now` as a low-risk systems improvement |
| `#1552` | no score yet | 0 | pass embeddings + RecurLoRA on recurrence passes | Medium. Implementation described, no run evidence. | Medium-high | `test early` after a stable recurrence base exists |

### New high-priority watchlist beyond the original four

| PR | Claimed result | Why it matters | Risk | Decision |
|---|---:|---|---|---|
| `#1576` | `1.01671233 BPB` | Best claimed overall result; new GDN-Hybrid Track A line | Very high. Architecture pivot and hard to reconcile with the current Track B plan quickly. | `defer until base is stable` |
| `#1578` | `1.06681663 BPB` | Huge tokenizer-only gain on top of `#1529` | High. Requires full retokenization and has legality / review uncertainty around casefold normalization. | `defer until base is stable` |
| `#1530` | `1.07336388 BPB` | The underlying VarLen + fused MLP + doc-TTT stack that `#1560` tunes | Medium | `must adopt now` as the main implementation reference for post-`#1493` Track B |
| `#1523` | `1.0778 BPB` | Introduces triple recurrence, banking, fused MLP, Muon 0.97 | Medium | `test early` only where pieces remain useful after `#1530/#1560` |

## Delta vs April 9 merged record (`#1493`)

### What looks broadly useful

- Faster data / compute path:
  - `#1532` async loader
  - `#1530` varlen attention
  - `#1530` fused MLP
- Better short-run schedule tuning:
  - `#1560` warmdown fraction `0.75`
  - `#1560` TTT chunk size `48`
  - `#1530/#1560` Muon momentum around `0.97`
- Recurrence refinements:
  - `#1523` / `#1518` wider recurrence over layers `3-5`
  - `#1518` per-pass embeddings
- Decoder routing:
  - `#1518` asymmetric two-lane routing is promising, but not the first thing to port

### What looks valuable but should wait

- `#1518` Tap-In V6:
  custom eval-time matcher, strong claimed benefit, but adds review and implementation surface
- `#1552` RecurLoRA:
  plausible low-byte extension, but no run evidence yet
- `#1578` casefold tokenizer:
  large upside, but requires dataset/tokenizer rebuild and may be rule-sensitive
- `#1576` GDN-Hybrid:
  strongest claim overall, but effectively a new project branch

## Ranked target-stack decision

### `must adopt now`

1. Treat `#1530/#1560` as the primary Track B target family, not `#1493`
2. Port the `#1532` loader ideas into the readable frontier script
3. Preserve the readable April SP8192 base only as the starting surface, not as the target architecture

### `test early`

1. Wider recurrence over `3-5`
2. Per-pass embeddings from `#1518` and `#1552`
3. Asymmetric two-lane decoder routing from `#1518`

### `defer until base is stable`

1. Casefold tokenizer path from `#1578`
2. GDN-Hybrid Track A path from `#1576`
3. Tap-In V6 custom matcher from `#1518`

### `ignore` for the current project branch

1. Off-frontier SP1024 / non-record architecture pivots
2. PRs without multi-seed evidence or without a credible path to the current frontier

## Updated project target

- **Best credible overall pending claim:** `#1576` at `1.01671233 BPB` (Track A, architecture pivot)
- **Best credible Track B same-family claim:** `#1518` at `1.073938 BPB`
- **Best low-friction same-family implementation target:** `#1560/#1530`

Practical project stance:
- keep the mainline effort on the readable SP8192 Track B family
- watch `#1576` as a separate architecture branch, not as the immediate implementation target
