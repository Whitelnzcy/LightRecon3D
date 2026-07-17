#!/usr/bin/env python3
"""Build an auditable Python 3.11 dependency set for Plane-DUSt3R.

The official Plane-DUSt3R repository combines current MASt3R requirements
with the much older NonCuboidRoom requirements.  Installing those files
verbatim can replace the README-pinned PyTorch build and asks Python 3.11 to
build packages that only supported Python 3.6.  This tool preserves every
unaffected requirement, removes only explicitly controlled packages, and
writes both their compatible pins and a machine-readable audit record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


REQUIREMENT_FILES = (
    "MASt3R/requirements.txt",
    "MASt3R/dust3r/requirements.txt",
    "NonCuboidRoom/requirements.txt",
)

# PyTorch stays installed by conda exactly as requested by the Plane-DUSt3R
# README.  The remaining versions have Python 3.11 wheels while staying close
# to the APIs used by the old NonCuboidRoom code.
COMPATIBILITY_PINS = {
    "torch": "torch==2.2.0",
    "torchvision": "torchvision==0.17.0",
    "torchaudio": "torchaudio==2.2.0",
    "numpy": "numpy==1.26.4",
    "scipy": "scipy==1.11.4",
    "opencv-python": "opencv-python==4.10.0.84",
    "shapely": "Shapely==1.8.5.post1",
    "matplotlib": "matplotlib==3.7.5",
    "mmcv": "mmcv==1.7.2",
    "easydict": "easydict==1.13",
    "numba": "numba==0.59.1",
    "pillow": "Pillow==9.5.0",
    "pyyaml": "PyYAML==6.0.2",
    "tensorboardx": "tensorboardX==2.6.2.2",
    "roma": "roma==1.5.6",
}

CONDA_MANAGED = frozenset({"torch", "torchvision", "torchaudio"})
UNSAFE_PREFIXES = (
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-e",
    "--editable",
)
NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    lowered = stripped.lower()
    if any(
        lowered == prefix or lowered.startswith(prefix + " ")
        for prefix in UNSAFE_PREFIXES
    ):
        raise ValueError(f"nested or editable requirement is not allowed: {line!r}")
    if stripped.startswith("-"):
        raise ValueError(f"unsupported pip option in requirements: {line!r}")
    match = NAME_RE.match(stripped)
    if match is None:
        raise ValueError(f"cannot determine requirement name: {line!r}")
    return normalize_name(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_lines(lines: Iterable[str], source: str) -> tuple[list[str], list[dict]]:
    kept: list[str] = []
    replacements: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        package = requirement_name(line)
        if package in COMPATIBILITY_PINS:
            replacements.append(
                {
                    "source": source,
                    "line_number": line_number,
                    "official_requirement": line.strip(),
                    "normalized_package": package,
                    "replacement": COMPATIBILITY_PINS[package],
                    "installer": "conda" if package in CONDA_MANAGED else "pip",
                }
            )
            continue
        kept.append(line)
    return kept, replacements


def write_outputs(official_repo: Path, output_dir: Path) -> dict:
    official_repo = official_repo.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    files: list[dict] = []
    all_replacements: list[dict] = []

    for relative in REQUIREMENT_FILES:
        source = official_repo / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing official requirements file: {source}")
        original = source.read_text(encoding="utf-8").splitlines(keepends=True)
        kept, replacements = sanitize_lines(original, relative)
        output_name = relative.replace("/", "__")
        destination = output_dir / output_name
        destination.write_text("".join(kept), encoding="utf-8", newline="\n")
        files.append(
            {
                "source": str(source),
                "source_relative": relative,
                "source_sha256": sha256(source),
                "sanitized": str(destination),
                "original_lines": len(original),
                "kept_lines": len(kept),
                "replaced_lines": len(replacements),
            }
        )
        all_replacements.extend(replacements)

    constraints = output_dir / "constraints.txt"
    constraints.write_text(
        "\n".join(COMPATIBILITY_PINS.values()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    pip_pins = output_dir / "python311_compatibility.txt"
    pip_pins.write_text(
        "\n".join(
            pin
            for package, pin in COMPATIBILITY_PINS.items()
            if package not in CONDA_MANAGED
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    audit = {
        "schema_version": 1,
        "kind": "plane_dust3r_python311_requirements",
        "official_repo": str(official_repo),
        "files": files,
        "compatibility_pins": COMPATIBILITY_PINS,
        "conda_managed": sorted(CONDA_MANAGED),
        "replacements": all_replacements,
        "constraints": str(constraints),
        "python311_compatibility": str(pip_pins),
    }
    audit_json = output_dir / "requirements_audit.json"
    audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    markdown = [
        "# Plane-DUSt3R Python 3.11 requirement audit",
        "",
        "The official requirement files are preserved except for the controlled packages below.",
        "PyTorch is installed by conda; old NonCuboidRoom pins are replaced with Python 3.11 compatible versions.",
        "",
        "| Source | Official requirement | Controlled replacement | Installer |",
        "|---|---|---|---|",
    ]
    for item in all_replacements:
        markdown.append(
            f"| `{item['source']}:{item['line_number']}` | "
            f"`{item['official_requirement']}` | `{item['replacement']}` | "
            f"{item['installer']} |"
        )
    (output_dir / "requirements_audit.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8", newline="\n"
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_repo", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit = write_outputs(args.official_repo, args.output_dir)
    print(
        json.dumps(
            {
                "files": len(audit["files"]),
                "controlled_requirements": len(audit["replacements"]),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
