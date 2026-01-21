from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def run(command: List[str], cwd: Path | None = None) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def iter_c_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.c")):
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


def _graphson_value(value: object) -> object:
    if isinstance(value, dict) and "@value" in value:
        return value["@value"]
    return value


def _graphson_map(value: object) -> Dict[str, object]:
    value = _graphson_value(value)
    if isinstance(value, list):
        mapped: Dict[str, object] = {}
        for i in range(0, len(value), 2):
            if i + 1 < len(value):
                mapped[str(value[i])] = value[i + 1]
        return mapped
    if isinstance(value, dict):
        return value
    return {}


def _graphson_list(value: object) -> List[object]:
    value = _graphson_value(value)
    if isinstance(value, list):
        return value
    return []


def _normalize_props(props: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in _graphson_map(props).items():
        value = _graphson_value(value)
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
    graph = _graphson_map(_graphson_value(data))
    if isinstance(graph, dict) and "graph" in graph:
        graph = _graphson_map(graph["graph"])
    vertices = _graphson_list(graph.get("vertices", []))
    edges = _graphson_list(graph.get("edges", []))

    nodes: List[Dict[str, object]] = []
    id_to_index: Dict[object, int] = {}

    for idx, node in enumerate(vertices):
        node = _graphson_map(node)
        node_id = _graphson_value(node.get("id"))
        label = _graphson_value(node.get("label", "UNKNOWN"))
        props = _normalize_props(node.get("properties", {}) if isinstance(node, dict) else {})
        code = props.get("CODE") or props.get("code") or label
        nodes.append({"id": node_id, "code": code, "label": label})
        id_to_index[node_id] = idx

    edge_list: List[Tuple[int, int, str]] = []
    for edge in edges:
        edge = _graphson_map(edge)
        src = _graphson_value(edge.get("outV"))
        dst = _graphson_value(edge.get("inV"))
        rel = _graphson_value(edge.get("label", "UNKNOWN"))
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


def _process_file(
    c_file: Path,
    joern_parse: Path,
    joern_export: Path,
    tmp_root: Path,
    index: int,
) -> Dict[str, object] | None:
    tmp_dir = tmp_root / f"joern_{index}"
    export_json = extract_graphson(c_file, joern_parse, joern_export, tmp_dir)
    entry = build_jsonl_entry(c_file, export_json)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return entry


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

    c_files = list(iter_c_files(input_dir))
    if args.limit:
        c_files = c_files[: args.limit]

    if args.workers <= 0:
        args.workers = max(os.cpu_count() or 1, 1)

    processed: set[str] = set()
    write_mode = "w"
    if args.resume and output_jsonl.exists():
        write_mode = "a"
        with output_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    processed.add(json.loads(line).get("path", ""))
                except json.JSONDecodeError:
                    continue
        c_files = [path for path in c_files if str(path) not in processed]

    with output_jsonl.open(write_mode, encoding="utf-8") as handle:
        if args.workers <= 1:
            for idx, c_file in enumerate(c_files):
                entry = _process_file(c_file, joern_parse, joern_export, tmp_root, idx)
                if entry is None:
                    continue
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                entries = executor.map(
                    _process_file,
                    c_files,
                    [joern_parse] * len(c_files),
                    [joern_export] * len(c_files),
                    [tmp_root] * len(c_files),
                    list(range(len(c_files))),
                    chunksize=max(args.chunksize, 1),
                )
                for entry in entries:
                    if entry is None:
                        continue
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joern-home", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--tmp-root", default="/tmp/joern_cpg")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of C files for a quick smoke test.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for Joern extraction (<=0 uses CPU count).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunk size for parallel extraction scheduling.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing JSONL and skip already processed paths.",
    )
    extract_dataset(parser.parse_args())
