#!/usr/bin/env python3
"""Download and extract dataset PDF archives into the expected data folders."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASE_URL = (
    "https://media.githubusercontent.com/media/"
    "Der********/agentic_reproducibility_assessment/main"
)


@dataclass(frozen=True)
class ArchiveSpec:
    dataset_dir: str
    zip_name: str
    extracted_dir: str

    @property
    def relative_zip_path(self) -> str:
        return f"data/{self.dataset_dir}/{self.zip_name}"


ARCHIVES = (
    ArchiveSpec(
        dataset_dir="reprobench",
        zip_name="reprobench_papers_original.zip",
        extracted_dir="papers_original",
    ),
    ArchiveSpec(
        dataset_dir="reprobench",
        zip_name="reprobench_papers_humanreplication.zip",
        extracted_dir="papers_humanreplication",
    ),
    ArchiveSpec(
        dataset_dir="reproscreener",
        zip_name="reproscreener_papers_original.zip",
        extracted_dir="papers_original",
    ),
    ArchiveSpec(
        dataset_dir="resciencec",
        zip_name="resciencec_papers_original.zip",
        extracted_dir="papers_original",
    ),
    ArchiveSpec(
        dataset_dir="resciencec",
        zip_name="resciencec_papers_humanreplication.zip",
        extracted_dir="papers_humanreplication",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the tracked PDF zip archives and extract them into the "
            "expected data/<dataset>/papers_* folders."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            "Base URL that hosts the repository files. Defaults to the GitHub "
            "media URL for the main branch."
        ),
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Local data directory root relative to the repository root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Redownload archives and recreate extracted papers_* folders even if "
            "they already exist."
        ),
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_archive(archive_path: Path, destination_dir: Path, extracted_dir_name: str) -> None:
    extracted_dir = destination_dir / extracted_dir_name
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        expected_prefix = f"{extracted_dir_name}/"
        if not any(name == expected_prefix or name.startswith(expected_prefix) for name in names):
            raise ValueError(
                f"{archive_path} does not contain the expected top-level folder "
                f"{expected_prefix!r}."
            )
        archive.extractall(destination_dir)


def process_archive(spec: ArchiveSpec, base_url: str, data_root: Path, force: bool) -> None:
    dataset_dir = data_root / spec.dataset_dir
    archive_path = dataset_dir / spec.zip_name
    extracted_dir = dataset_dir / spec.extracted_dir
    archive_url = f"{base_url.rstrip('/')}/{spec.relative_zip_path}"

    if force or not archive_path.exists():
        action = "Downloading" if not archive_path.exists() else "Redownloading"
        print(f"{action} {archive_url} -> {archive_path}")
        download_file(archive_url, archive_path)
    else:
        print(f"Using existing archive {archive_path}")

    if force or not extracted_dir.exists():
        action = "Extracting" if not extracted_dir.exists() else "Re-extracting"
        print(f"{action} {archive_path} -> {dataset_dir}")
        extract_archive(archive_path, dataset_dir, spec.extracted_dir)
    else:
        print(f"Skipping extraction for existing folder {extracted_dir}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = (repo_root / args.data_root).resolve()

    try:
        for spec in ARCHIVES:
            process_archive(spec, args.base_url, data_root, args.force)
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
