# Frontier Workspace

This directory is the active leaderboard workspace.

Goals:
- keep a readable CUDA development surface separate from packed submission wrappers
- keep frontier triage and experiment notes in-repo
- keep temporary run artifacts out of the repo root

Layout:
- `cuda/train_gpt_frontier.py`: canonical readable CUDA development script
- `mlx/`: local Apple Silicon proxy scripts for strong leaderboard architectures
- `docs/`: upstream triage, stack decisions, and execution notes
- `docs/compliance_checklist.md`: project-local checklist derived from issue `#1017`
- `docs/workflow.md`: default local -> cloud -> submission workflow
- `docs/run_intake_template.md`: copy/paste template for requesting experiments and promotions
- `docs/research_backlog.md`: current paper- and PR-driven research priorities
- `docs/submission_template.md`: template for final leaderboard-style README
- `logs/lab_notebook.md`: running experiment notebook
- `logs/experiments.tsv`: leaderboard-style experiment ledger for scored runs
- `cloud/candidate_pack.md`: exact or near-exact 8xH100 candidate commands and gates
- `workdirs/`: untracked outputs for local and cloud runs

Rules:
- do not edit the top-level baseline scripts unless there is a deliberate reason
- do not use packed LZMA wrappers as the development source of truth
- log each meaningful experiment batch before promoting it to cloud
- append every scored run to `logs/experiments.tsv` with loss, BPB, and artifact size
- treat `docs/compliance_checklist.md` as mandatory before trusting any score or writeup
- use `docs/workflow.md` as the default operating process unless explicitly overridden
