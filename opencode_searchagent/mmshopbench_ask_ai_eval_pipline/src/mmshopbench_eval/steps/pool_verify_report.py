

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mmshopbench_eval.steps.pool_verify_common import read_jsonl, s, split_ids_field

AXES = ("axis_a", "axis_b", "axis_c")
AXIS_ORDER = {
    "axis_a": ("text_only", "mixed", "visual_only", "unknown"),
    "axis_b": ("text_dominant", "mixed", "visual_dominant"),
    "axis_c": ("text_sufficient", "mixed", "image_required"),
}


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-results", required=True, help="注入池版判官 judge_results.jsonl")
    ap.add_argument("--meta", required=True, help="prepare 步的 *_meta.jsonl")
    ap.add_argument("--axis-labels", default="", help="axis_labels.jsonl(A/B/C)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta_by_cid = {s(m.get("conversation_id")): m for m in read_jsonl(args.meta)}
    axis_by_cid: dict[str, dict[str, Any]] = {}
    if args.axis_labels:
        axis_by_cid = {s(a.get("conversation_id")): a for a in read_jsonl(args.axis_labels)}

    out_rows: list[dict[str, Any]] = []
    for jr in read_jsonl(args.judge_results):
        cid = s(jr.get("sample_id"))
        m = meta_by_cid.get(cid)
        if not m:
            continue
        if not s(jr.get("judge_success", "")) and s(jr.get("judge_error")):
            continue
        sat_ids = set(split_ids_field(jr.get("满足要求的商品id")))
        union_ids = [s(x) for x in m.get("union_ids") or []]
        final_ids = set(s(x) for x in m.get("final_ids") or [])
        sat_in_union = [i for i in union_ids if i in sat_ids]
        pool_any = 1 if sat_in_union else 0
        final_any = 1 if (sat_ids & final_ids) else 0
        ax = axis_by_cid.get(cid, {})
        out_rows.append({
            "conversation_id": cid,
            "pool_satisfied_any": pool_any,
            "final_satisfied_any": final_any,
            "n_union": len(union_ids),
            "n_final": len(final_ids),
            "satisfied_ids": sat_in_union,
            "union_ids": union_ids,
            "final_ids": sorted(final_ids),
            "axis_a": s(ax.get("axis_a")),
            "axis_b": s(ax.get("axis_b")),
            "axis_c": s(ax.get("axis_c")),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out_rows)
    pool_r = _rate(out_rows, "pool_satisfied_any")
    final_r = _rate(out_rows, "final_satisfied_any")
    print(f"\n=== retrieval/verify 分解 (N={n}) ===", file=sys.stderr)
    print(f"  pool  satisfied@any (retrieval 上界): {pool_r:.1%}", file=sys.stderr)
    print(f"  final satisfied@any (verify 后)     : {final_r:.1%}", file=sys.stderr)
    print(f"  gap = pool - final (verify 损失)     : {pool_r - final_r:+.1%}", file=sys.stderr)
    bad = sum(1 for r in out_rows if r["final_satisfied_any"] > r["pool_satisfied_any"])
    if bad:
        print(f"  [warn] {bad} 条 final>pool(应为 0,检查并集/归属)", file=sys.stderr)

    for axis in AXES:
        print(f"\n=== 按 {axis} 切 pool/final/gap ===", file=sys.stderr)
        by: dict[str, list[dict[str, Any]]] = {}
        for r in out_rows:
            by.setdefault(r[axis] or "NA", []).append(r)
        for lab in AXIS_ORDER[axis] + ("NA",):
            g = by.get(lab, [])
            if g:
                pr, fr = _rate(g, "pool_satisfied_any"), _rate(g, "final_satisfied_any")
                print(f"  {lab:<15} n={len(g):<4} pool={pr:.1%}  final={fr:.1%}  gap={pr-fr:+.1%}", file=sys.stderr)

    print(f"\nwritten: {args.out} ({n} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
