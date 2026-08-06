#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_rows(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"无法解析输入 json 结构: {type(data)}")


def split_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        parts = [str(x) for x in raw]
    else:
        parts = re.split(r"[,\uFF0C\u3001;\s]+", str(raw))
    return [p.strip() for p in parts if p and p.strip()]


def compute_step1(rows: list[dict]) -> tuple[int, int]:
    total = len(rows)
    succ = sum(1 for r in rows if not (r.get("error") or "").strip())
    return succ, total


def compute_id_match(rows: list[dict]) -> tuple[int, int]:
    total = hit = 0
    for r in rows:
        ids = split_ids(r.get("Manual_annotation_item_id"))
        pred = r.get("pred_model_output")
        if not ids or pred is None:
            continue
        total += 1
        pred_str = pred if isinstance(pred, str) else json.dumps(pred, ensure_ascii=False)
        if any(i in pred_str for i in ids):
            hit += 1
    return hit, total


def read_report(path: str) -> dict:
    p = Path(path) if path else None
    if not p or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return sum(int(r.get(key) or 0) for r in rows) / len(rows)


def agg_egvs(egvs_jsonl: str) -> dict:
    
    p = Path(egvs_jsonl)
    if not egvs_jsonl or not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("err") or r.get("err") == "verify_parse_fail"]
    gt_ok = [r for r in ok if r.get("has_gt")]
    gt_all = [r for r in rows if r.get("has_gt")]
    tot = len(rows)
    nev = sum(1 for r in rows if r.get("err") == "no_evidence")

    def eff_j(r):
        return int(r.get("baseline_final_any") or 0) if r.get("err") == "no_evidence" else int(r.get("egvs_final_any") or 0)

    def eff_i(r):
        return int(r.get("baseline_id_hit") or 0) if r.get("err") == "no_evidence" else int(r.get("egvs_id_hit") or 0)

    def frac(rs, fn):
        return (sum(fn(r) for r in rs) / len(rs)) if rs else None

    gi = lambda k: (lambda r: int(r.get(k) or 0))
    return {
        "n_total": tot, "n_ok": len(ok), "nev": nev,
        "fallback": sum(int(r.get("fallback") or 0) for r in ok), "n_gt": len(gt_all),
        "jN": (frac(ok, gi("baseline_final_any")), frac(ok, gi("egvs_final_any")), frac(ok, gi("pool_any"))),
        "idN": (frac(gt_ok, gi("baseline_id_hit")), frac(gt_ok, gi("egvs_id_hit")), frac(gt_ok, gi("pool_id_hit"))),
        "jA": (frac(rows, gi("baseline_final_any")), frac(rows, eff_j), frac(rows, gi("pool_any"))),
        "idA": (frac(gt_all, gi("baseline_id_hit")), frac(gt_all, eff_i), frac(gt_all, gi("pool_id_hit"))),
    }


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _tri(base: float | None, egvs: float | None, orc: float | None) -> str:
    if base is None and egvs is None and orc is None:
        return "-"
    d = (egvs - base) if (egvs is not None and base is not None) else None
    dpart = f"(Δ{d * 100:+.1f})" if d is not None else ""
    return f"{_pct(base)}→{_pct(egvs)}{dpart}/{_pct(orc)}"


def main() -> None:
    ap = argparse.ArgumentParser(description="追加一次 egvs run 的指标到独立汇总文件")
    ap.add_argument("--output-json", required=True, help="step1 生成的 output.json")
    ap.add_argument("--egvs-jsonl", default="", help="egvs_self_verify 产出的 egvs.jsonl")
    ap.add_argument("--orig-judge-report", default="", help="对原始卡跑 offline judge 的 judge_report.json(直评 llm_judge)")
    ap.add_argument("--real-modelname", default="")
    ap.add_argument("--verify-model", default="", help="egvs self-verify 用的模型")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--run-ts", default="")
    ap.add_argument("--summary-md", required=True)
    ap.add_argument("--summary-csv", default="")
    args = ap.parse_args()

    rows = load_rows(args.output_json)
    s_succ, s_total = compute_step1(rows)
    s_rate = (s_succ / s_total) if s_total else 0.0
    id_hit, id_total = compute_id_match(rows)
    id_rate = (id_hit / id_total) if id_total else 0.0

    jrpt = read_report(args.orig_judge_report)
    j_hit_rate = jrpt.get("hit_rate")

    e = agg_egvs(args.egvs_jsonl)

    ds_name = Path(args.dataset).name if args.dataset else ""
    step1_cell = f"{s_succ}/{s_total}"
    llm_cell = "-" if j_hit_rate is None else f"{j_hit_rate:.3f}"
    id_cell = "-" if not id_total else f"{id_rate:.3f}"

    def _c(k):
        return _tri(*e[k]) if e.get(k) else "-"

    jN_cell, jA_cell, idN_cell, idA_cell = _c("jN"), _c("jA"), _c("idN"), _c("idA")
    n_cell = f"{e['n_ok']}(nev{e['nev']})" if e else "-"

    print("=" * 68)
    print("[egvs 汇总] 本次 run")
    print(f"  real_modelname : {args.real_modelname}   verify: {args.verify_model}   数据集: {ds_name}")
    print(f"  step1 成功率   : {step1_cell}")
    print(f"  [直评] llm_judge / id文本 : {llm_cell} / {id_cell}")
    print(f"  [egvs] 判  @N : {jN_cell}    @全289: {jA_cell}")
    print(f"  [egvs] id  @N : {idN_cell}    @全289: {idA_cell}")
    print("=" * 68)

    hdr = ["时间", "real_modelname", "verifier", "step1", "直评judge", "id文本",
           "egvs判@N", "egvs判@全289", "egvs-id@N", "egvs-id@全289", "N(nev)"]
    cells = [args.run_ts, args.real_modelname, args.verify_model, step1_cell,
             llm_cell, id_cell, jN_cell, jA_cell, idN_cell, idA_cell, n_cell]

    md_path = Path(args.summary_md)
    new_file = not md_path.exists()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("# EGVS + 直评 统一实验汇总\n\n")
            f.write("> 直评judge/id文本=baseline同口径(判官失败已排除);egvs判/id=注入池+self-verify重选口径。\n")
            f.write("> **egvs 单元格 = `baseline→EGVS(Δ)/oracle`**:baseline=模型原始推荐命中率、EGVS=self-verify重选后命中率(Δ=相对baseline提升)、oracle=候选池(top20∪final)命中率=检索上界。\n")
            f.write("> **两种口径**:egvs判=判官判满足全部硬约束;egvs-id=命中GT Manual_annotation_item_id(与判官无关)。\n")
            f.write("> **@N**=可核验子集(289−no_evidence),各模型N不同不可横比;**@全289**=固定分母289(no_evidence计miss、EGVS回退=baseline),可横比。\n\n")
            f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        f.write("| " + " | ".join(str(x) for x in cells) + " |\n")
    print(f"==> 已追加到文档: {md_path}")

    if args.summary_csv:
        import csv
        csv_path = Path(args.summary_csv)
        new_csv = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if new_csv:
                w.writerow(hdr)
            w.writerow(cells)
        print(f"==> 已追加到表格: {csv_path}")


if __name__ == "__main__":
    main()
