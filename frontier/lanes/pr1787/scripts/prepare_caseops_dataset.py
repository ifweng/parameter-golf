#!/usr/bin/env python3
"""Download the prebuilt CaseOps dataset used by the PR #1787 lane.

This prepares the cloud-side data cache. It is not part of the timed training
or eval run. The output layout intentionally matches train_gpt.py's default
CASEOPS_ENABLED=1 path:

  data/datasets/fineweb10B_sp8192_caseops/datasets/
    datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
      fineweb_train_*.bin
      fineweb_val_000000.bin
      fineweb_val_bytes_000000.bin
    tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "romeerp/parameter-golf-caseops-v1"
DATASET_DIR = "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
TOKENIZER_DIR = "datasets/tokenizers"
TOKENIZER_MODEL = "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
TOKENIZER_VOCAB = "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.vocab"


def materialize(repo_id: str, filename: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached_path = Path(
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    ).resolve(strict=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(cached_path, tmp)
    except OSError:
        shutil.copy2(cached_path, tmp)
    tmp.replace(destination)


def expected_files(train_shards: int) -> list[str]:
    files = [
        "datasets/manifest.json",
        "datasets/tokenizer_config.export.json",
        f"{TOKENIZER_DIR}/{TOKENIZER_MODEL}",
        f"{TOKENIZER_DIR}/{TOKENIZER_VOCAB}",
        f"{DATASET_DIR}/fineweb_val_000000.bin",
        f"{DATASET_DIR}/fineweb_val_bytes_000000.bin",
    ]
    for i in range(train_shards):
        files.append(f"{DATASET_DIR}/fineweb_train_{i:06d}.bin")
    return files


def verify_available(repo_id: str, files: list[str]) -> None:
    available = set(HfApi().list_repo_files(repo_id, repo_type="dataset"))
    missing = [path for path in files if path not in available]
    if missing:
        joined = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(f"repo {repo_id} is missing expected files:\n{joined}")


def verify_local(out_root: Path, train_shards: int) -> None:
    data_path = out_root / DATASET_DIR.removeprefix("datasets/")
    tokenizer_path = out_root / TOKENIZER_DIR.removeprefix("datasets/") / TOKENIZER_MODEL
    train_files = sorted(data_path.glob("fineweb_train_*.bin"))
    val_files = sorted(data_path.glob("fineweb_val_*.bin"))
    val_byte_files = sorted(data_path.glob("fineweb_val_bytes_*.bin"))
    if len(train_files) != train_shards:
        raise SystemExit(f"expected {train_shards} train shards, found {len(train_files)} in {data_path}")
    if len(val_files) != 1:
        raise SystemExit(f"expected 1 val shard, found {len(val_files)} in {data_path}")
    if len(val_byte_files) != 1:
        raise SystemExit(f"expected 1 val byte sidecar, found {len(val_byte_files)} in {data_path}")
    if not tokenizer_path.is_file():
        raise SystemExit(f"missing tokenizer model: {tokenizer_path}")
    print(f"CASEOPS_DATA_PATH={data_path}")
    print(f"TOKENIZER_PATH={tokenizer_path}")
    print(f"train_shards={len(train_files)} val_shards={len(val_files)} val_byte_sidecars={len(val_byte_files)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/datasets/fineweb10B_sp8192_caseops/datasets"),
        help="Output root matching train_gpt.py's CaseOps default.",
    )
    parser.add_argument("--train-shards", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download/materialization workers.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; only validate files already present at --out-root.",
    )
    parser.add_argument(
        "--check-remote-only",
        action="store_true",
        help="Validate that the expected files exist on Hugging Face, then exit.",
    )
    args = parser.parse_args()

    if args.train_shards < 1 or args.train_shards > 80:
        raise SystemExit("--train-shards must be between 1 and 80")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")

    files = expected_files(args.train_shards)
    if args.check_remote_only:
        verify_available(args.repo_id, files)
        print(f"remote OK: {args.repo_id} has {len(files)} expected files")
        return

    if not args.verify_only:
        verify_available(args.repo_id, files)
        jobs = []
        for filename in files:
            rel = Path(filename).relative_to("datasets")
            destination = args.out_root / rel
            jobs.append((filename, destination))
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(materialize, args.repo_id, filename, destination, force=args.force): (
                    idx,
                    filename,
                    destination,
                )
                for idx, (filename, destination) in enumerate(jobs, start=1)
            }
            for future in as_completed(futures):
                idx, filename, destination = futures[future]
                future.result()
                print(f"[{idx}/{len(files)}] {filename} -> {destination}", flush=True)

    verify_local(args.out_root, args.train_shards)


if __name__ == "__main__":
    main()
