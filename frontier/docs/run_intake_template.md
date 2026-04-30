# Run Intake Template

Use this when you want me to apply the default frontier workflow without extra back-and-forth.

## Copy/Paste Template

```text
Start local experiment:
- objective:
- branch or lane:
- baseline to compare against:
- change to test:
- success metric:
- constraints or notes:
```

```text
Promote this to cloud candidate:
- run_id:
- track:
- reason to promote:
- expected gain:
- budget or hardware notes:
- constraints or notes:
```

```text
Package this as a submission candidate:
- run_id:
- track:
- target leaderboard line to beat:
- required artifacts:
- notes for the writeup:
```

## Minimal Version

If you want to move fast, this is enough:

```text
Start local experiment: test <change> against <baseline>; success = <metric>
```

## What I Will Infer By Default

If you leave fields blank, I will usually infer:
- the active main lane unless you name a different branch
- the latest strong internal control as the comparison baseline
- the smallest reasonable local run first
- compliance review before any cloud promotion
- leaderboard-style logging for any meaningful scored run

## Good Examples

```text
Start local experiment:
- objective: check whether higher qk gain still helps on the SP8192 proxy
- branch or lane: main-control
- baseline to compare against: apr05_sp8192_proxy_local_pilot_v2
- change to test: qk_gain_init 4.0 -> 5.25
- success metric: lower val_bpb without artifact regression
- constraints or notes: keep runtime short and local-only
```

```text
Promote this to cloud candidate:
- run_id: main_control_local_ablation_003
- track: Track B
- reason to promote: local signal is positive and code path is stable
- expected gain: modest fixed-predictor improvement before TTT
- budget or hardware notes: prepare for RunPod, smoke first
- constraints or notes: verify compliance checklist before launch
```
