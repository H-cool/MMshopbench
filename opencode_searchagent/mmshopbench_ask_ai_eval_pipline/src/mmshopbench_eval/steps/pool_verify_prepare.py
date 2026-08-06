

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mmshopbench_eval.steps.pool_verify_common import (
    final_card_from_pred, make_pred_with_ids, pool_by_run_id, read_json, s, uniq,
)


def _rerun_search_pool(trace_path: str, top_k_per_query: int) -> dict[str, list[str]]:
    
    opencode_root = str(Path(__file__).resolve().parents[4])
    if opencode_root not in sys.path:
        sys.path.insert(0, opencode_root)
    from examples.tool_registry import platform_product_search, product_image_search
    from mmshopbench_eval.steps.pool_verify_common import _ids_from_tool_content, read_jsonl

    out: dict[str, list[str]] = {}
    for rec in read_jsonl(trace_path):
        rid = s(rec.get("run_id"))
        if not rid:
            continue
        ids: list[str] = []
        for st in rec.get("steps") or []:
            for msg in (st.get("output", {}) or {}).get("messages", []) or []:
                for tc in msg.get("tool_calls") or []:
                    name = s(tc.get("name")) or s((tc.get("function") or {}).get("name"))
                    args = tc.get("arguments") or (tc.get("function") or {}).get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    args = args if isinstance(args, dict) else {}
                    try:
                        if name == "platform_product_search":
                            q = args.get("query") or []
                            ids += _ids_from_tool_content(platform_product_search(q if isinstance(q, list) else [q]))
                        elif name == "product_image_search":
                            ids += _ids_from_tool_content(
                                product_image_search(s(args.get("image_url")), s(args.get("region")))
                            )
                    except Exception as exc:
                        print(f"rerun search failed rid={rid}: {exc}", file=sys.stderr)
        out[rid] = uniq(ids)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True, help="Step1 输出 output.json")
    ap.add_argument("--trace-jsonl", required=True, help="Step1 输出 *-trace.jsonl")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--topn", type=int, default=20, help="候选池截断(每会话)")
    ap.add_argument("--rerun-search", action="store_true", help="拿不到 trace 结果时,重放搜索工具兜底")
    ap.add_argument("--text-top-k", type=int, default=10)
    args = ap.parse_args()

    data = read_json(args.input_json)
    rows = data if isinstance(data, list) else data.get("data", data)

    pool_by_rid = (
        _rerun_search_pool(args.trace_jsonl, args.text_top_k)
        if args.rerun_search
        else pool_by_run_id(args.trace_jsonl)
    )

    injected: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    n_empty_pool = 0
    dropped = 0
    for r in rows:
        cid = s(r.get("conversation_id"))
        rid = s(r.get("run_id"))
        final_ids, img_idx = final_card_from_pred(s(r.get("pred_model_output")))
        pool_full = uniq(pool_by_rid.get(rid, []))
        pool = pool_full[: args.topn]
        if not pool:
            n_empty_pool += 1
        dropped += max(0, len(pool_full) - args.topn)
        union = uniq(pool + final_ids)

        rec = dict(r)
        rec["pred_model_output"] = make_pred_with_ids(union, img_idx)
        injected.append(rec)
        meta.append({
            "conversation_id": cid, "run_id": rid,
            "pool_ids": pool, "n_pool_full": len(pool_full),
            "final_ids": final_ids, "union_ids": union,
        })

    inj_path = Path(f"{args.out_prefix}_pool_injected.json")
    meta_path = Path(f"{args.out_prefix}_meta.jsonl")
    inj_path.parent.mkdir(parents=True, exist_ok=True)
    inj_path.write_text(json.dumps(injected, ensure_ascii=False, indent=2), encoding="utf-8")
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    max_union = max((len(m["union_ids"]) for m in meta), default=0)
    print(f"records={len(injected)}  empty_pool={n_empty_pool}  dropped(>topn)={dropped}", file=sys.stderr)
    print(f"max union size={max_union}  -> 判官需 --max-card-items-for-judge >= {max_union}", file=sys.stderr)
    print(f"injected: {inj_path}", file=sys.stderr)
    print(f"meta:     {meta_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
