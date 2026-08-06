#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算 pred_model_output 命中 Manual_annotation_item_id 的平均分")
    parser.add_argument("--input-json", required=True, help="第一步 agent 生成的 output json 路径")
    return parser.parse_args()


def load_rows(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("results", "data", "rows", "records"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"无法解析输入 json 结构: {type(data)}")


def split_ids(raw: Any) -> List[str]:
    
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        parts = [str(x) for x in raw]
    else:
        parts = re.split(r"[,\uFF0C\u3001;\s]+", str(raw))
    return [p.strip() for p in parts if p and p.strip()]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_json)

    total = 0
    hit = 0
    skipped = 0
    for row in rows:
        ids = split_ids(row.get("Manual_annotation_item_id"))
        pred = row.get("pred_model_output")
        if not ids or pred is None:
            skipped += 1
            continue
        pred_str = pred if isinstance(pred, str) else json.dumps(pred, ensure_ascii=False)
        total += 1
        if any(item_id in pred_str for item_id in ids):
            hit += 1

    score = (hit / total) if total else 0.0

    print("=" * 60)
    print("[item_id 命中率评分] pred_model_output 是否包含 Manual_annotation_item_id")
    print(f"  输入文件      : {args.input_json}")
    print(f"  参与评分样本数: {total}（跳过 {skipped} 条：无 id 或无 pred_model_output）")
    print(f"  命中样本数    : {hit}")
    print(f"  平均分        : {score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
