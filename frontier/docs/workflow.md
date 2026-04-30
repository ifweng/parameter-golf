# Project Workflow

This is the default operating workflow for this project.

Goals:
- move fast on local hardware
- only spend cloud money on high-signal candidates
- keep every serious result compliance-reviewable
- make final submission packaging incremental instead of last-minute

Read together with:
- [Submission Compliance Checklist](/Users/ifengwu/Projects/parameter-golf/frontier/docs/compliance_checklist.md)
- [Cloud Candidate Pack](/Users/ifengwu/Projects/parameter-golf/frontier/cloud/candidate_pack.md)
- [Run Intake Template](/Users/ifengwu/Projects/parameter-golf/frontier/docs/run_intake_template.md)

## Stage 1: Local Mac screening

Purpose:
- cheap directional testing
- kill weak ideas quickly
- validate code paths before cloud

Use local for:
- MLX proxy experiments
- architecture ablations
- recurrence and residual-topology checks
- tokenizer/accounting sanity checks
- rough artifact-size direction

Do not treat local as authoritative for:
- final leaderboard BPB
- CUDA throughput
- full TTT timing
- final quantization conclusions

Required output for every meaningful local run:
- one row appended to `frontier/logs/experiments.tsv`
- one short entry in `frontier/logs/lab_notebook.md`
- explicit track declaration: `Track A` or `Track B`
- status label: `positive`, `neutral`, `negative`, or `invalid`

Promotion gate from local to cloud:
- the result improves a target metric or meaningfully reduces risk
- the eval path is not obviously non-compliant
- the change is understandable enough to explain in a future README

## Stage 2: Cloud GPU validation

Purpose:
- run real score-bearing experiments
- measure runtime, artifact size, and full validation behavior

Cloud progression:
1. `smoke`
- verify code path, logging, artifact path, and eval path

2. `candidate`
- full-budget run on the intended track

3. `confirmation`
- 3-seed or more only after a candidate is competitive

Required checks before cloud promotion:
- track declared
- compliance checklist reviewed
- full validation path used
- artifact plausibly under `16,000,000` bytes
- expected train/eval time plausible under `600s + 600s`

## Stage 3: Submission packaging

Start submission packaging as soon as a candidate looks real.

For every serious candidate, keep:
- exact command
- exact code snapshot
- exact tokenizer / dataset variant
- train logs
- eval logs
- artifact size
- seed list
- compliance notes

Final submission folder should contain:
- `train_gpt.py`
- `README.md`
- `submission.json`
- training logs for required seeds
- `requirements.txt`

Use the template in:
- [submission_template.md](/Users/ifengwu/Projects/parameter-golf/frontier/docs/submission_template.md)

## Compliance gates

Before trusting a score:
- check [compliance_checklist.md](/Users/ifengwu/Projects/parameter-golf/frontier/docs/compliance_checklist.md)

Before promoting a candidate:
- state the track
- describe eval-time state changes
- confirm score-before-update if adaptive
- confirm single-pass scoring
- confirm full-vocab normalization
- confirm full validation set and correct byte accounting

Before submission:
- convert those answers into a README compliance section

## Suggested way to work with Codex

When you want to start a new line of work, give one of these commands:

- `Start local experiment: ...`
- `Promote this to cloud candidate: ...`
- `Package this as a submission candidate`

If you want a structured format, use:
- [run_intake_template.md](/Users/ifengwu/Projects/parameter-golf/frontier/docs/run_intake_template.md)

For local work, I will default to:
- using the smallest reasonable experiment
- updating the notebook and ledger
- calling out whether the result is only directional

For cloud work, I will default to:
- checking compliance first
- keeping commands reproducible
- narrowing to a small candidate set before asking for expensive runs

For submission work, I will default to:
- producing the required folder structure
- drafting a README in leaderboard style
- surfacing any missing artifact or compliance evidence

## Default branch structure

- `main-control`
- closest readable reproduction of the main merged line

- `main-ttt`
- same architecture, only TTT changes

- `main-quant`
- same architecture, only quantization changes

- `main-combined`
- combine proven TTT and quantization improvements only after each works separately

- tokenizer side branches
- always separate from main-line architecture work

## Default decision rule

- Local answers: `Is this worth pursuing?`
- Cloud answers: `Does it really work under the actual budget?`
- Submission answers: `Can we defend it line by line?`
