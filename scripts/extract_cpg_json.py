from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def run(command: List[str], cwd: Path | None = None) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def iter_c_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.c"):
        if path.is_file():
            yield path


def resolve_label(path: Path) -> int:
    parts = {part.lower() for part in path.parts}
    if "vul" in parts:
        return 1
    if "novul" in parts:
        return 0
    raise ValueError(f"Unable to infer label from path: {path}")


def extract_graphson(
    c_file: Path,
    joern_parse: Path,
    joern_export: Path,
    tmp_dir: Path,
) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for child in tmp_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)

    run([str(joern_parse), str(c_file)], cwd=tmp_dir)
    out_dir = tmp_dir / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    run([str(joern_export), "--repr=all", "--format=graphson", "--out", str(out_dir)], cwd=tmp_dir)

    export_json = out_dir / "export.json"
    if not export_json.exists():
        raise FileNotFoundError(f"export.json not found in {out_dir}")
    return export_json


def _normalize_props(props: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in (props or {}).items():
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict) and "value" in first:
                normalized[key] = first["value"]
            else:
                normalized[key] = first
        else:
            normalized[key] = value
    return normalized


def parse_graphson(export_json: Path) -> Tuple[List[Dict[str, object]], List[Tuple[int, int, str]]]:
    data = json.loads(export_json.read_text(encoding="utf-8"))
    graph = data.get("graph", data)
    vertices = graph.get("vertices", [])
    edges = graph.get("edges", [])

    nodes: List[Dict[str, object]] = []
    id_to_index: Dict[object, int] = {}

    for idx, node in enumerate(vertices):
        node_id = node.get("id")
        label = node.get("label", "UNKNOWN")
        props = _normalize_props(node.get("properties", {}) if isinstance(node, dict) else {})
        code = props.get("CODE") or props.get("code") or label
        nodes.append({"id": node_id, "code": code, "label": label})
        id_to_index[node_id] = idx

    edge_list: List[Tuple[int, int, str]] = []
    for edge in edges:
        src = edge.get("outV")
        dst = edge.get("inV")
        rel = edge.get("label", "UNKNOWN")
        if src not in id_to_index or dst not in id_to_index:
            continue
        edge_list.append((id_to_index[src], id_to_index[dst], rel))

    return nodes, edge_list


def build_jsonl_entry(
    c_file: Path,
    export_json: Path,
) -> Dict[str, object]:
    nodes, edges = parse_graphson(export_json)
    return {
        "path": str(c_file),
        "label": resolve_label(c_file),
        "nodes": nodes,
        "edges": edges,
    }


def extract_dataset(args: argparse.Namespace) -> None:
    joern_home = Path(args.joern_home)
    joern_parse = joern_home / "joern-parse"
    joern_export = joern_home / "joern-export"

    if not joern_parse.exists():
        raise FileNotFoundError(f"joern-parse not found at {joern_parse}")
    if not joern_export.exists():
        raise FileNotFoundError(f"joern-export not found at {joern_export}")

    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    tmp_root = Path(args.tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, c_file in enumerate(iter_c_files(input_dir)):
            tmp_dir = tmp_root / f"joern_{idx}"
            export_json = extract_graphson(c_file, joern_parse, joern_export, tmp_dir)
            entry = build_jsonl_entry(c_file, export_json)
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joern-home", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--tmp-root", default="/tmp/joern_cpg")
    extract_dataset(parser.parse_args())
