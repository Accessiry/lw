import json
import os
from datetime import datetime
from typing import Any, Dict


def make_run_dir(output_dir: str, run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
