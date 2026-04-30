# Submission README Template

Use this as the starting point for any serious candidate.

Reference style:
- [April 9 SP8192 leaderboard README](/Users/ifengwu/Projects/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md)

---

# Title: `<model name / technique summary>`

**val_bpb = `<mean>`** (`<seed-count>`-seed mean, std `<std>`) | **`<artifact size>` MB** | `8xH100 SXM`

## Result Table

| Seed | Pre-quant BPB | Post-quant BPB | Sliding BPB | Post-TTT BPB | Artifact |
|------|---------------:|----------------:|------------:|-------------:|---------:|
| ...  | ...            | ...             | ...         | ...          | ...      |

## Summary

Short paragraph explaining what changed and why it mattered.

## Key Techniques

1. Technique one
2. Technique two
3. Technique three

## Architecture

- tokenizer
- model width / depth
- recurrence schedule
- residual topology
- activation
- attention details

## Training

- optimizer
- LR / warmdown / EMA
- steps completed
- train runtime

## Quantization / Compression

- quantization method
- calibration method
- precision allocation
- compression method
- final artifact size

## Evaluation Track

State explicitly:
- `Track A` or `Track B`

## Compliance

Address the four conditions explicitly:

1. `Strict causal dependence`
- explain why prediction at token `t` only depends on the artifact and strict prefix

2. `Full normalized distribution`
- explain how the full vocab distribution is defined before scoring

3. `Score-before-update`
- explain exactly when scoring occurs and when state updates occur

4. `Single left-to-right pass`
- explain why there is no rescoring or second pass

## BPB Calculation

- explain how byte counting is done
- mention tokenizer lookup tables if relevant
- confirm full validation set and official ordering

## Runtime / Budget

- train time
- eval time
- artifact bytes

## Reproduction

Provide the exact command and required packages.

## Included Files

- `train_gpt.py`
- `README.md`
- `submission.json`
- `requirements.txt`
- logs

## Notes / Risks

- open questions
- any caveats
- anything the reviewer should verify carefully
