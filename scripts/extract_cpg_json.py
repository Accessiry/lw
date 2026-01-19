from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def extract_cpg(args: argparse.Namespace) -> None:
    joern_home = Path(args.joern_home)
    joern_parse = joern_home / "joern-parse"
    joern_export = joern_home / "joern-export"

    if not joern_parse.exists():
        raise FileNotFoundError(f"joern-parse not found at {joern_parse}")
    if not joern_export.exists():
        raise FileNotFoundError(f"joern-export not found at {joern_export}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpg_bin = output_dir / "code.cpg.bin"
    run([str(joern_parse), str(args.input_dir), "--out", str(cpg_bin)])
    run(
        [
            str(joern_export),
            "--repr",
            "cpg",
            "--format",
            "json",
            "--out",
            str(output_dir),
            str(cpg_bin),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joern-home", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    extract_cpg(parser.parse_args())
