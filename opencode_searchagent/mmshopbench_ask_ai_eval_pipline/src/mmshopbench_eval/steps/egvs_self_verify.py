

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("egvs")

from mmshopbench_eval.steps.pool_verify_common import (
    evidence_text_for, load_offline_detail, main_image_for, read_jsonl, s, uniq,
)
from mmshopbench_eval.steps.pool_verify_report import AXES, AXIS_ORDER, _rate
from mmshopbench_eval.steps.score_item_id_recall import split_ids

SYS = (
    "你是电商检索质量核验专家,依据商品标题、主图与文本详情(属性/SKU),"
    "严格逐条核验商品是否满足用户的全部硬性购物需求,只输出 JSON。"
)
INSTRUCTION = """用户购物需求(hard constraints,以 / 分隔,必须**全部**满足):
{attrs}
用户意图:{intent}

下面依次给出若干候选商品(标题+文本详情,可能附主图)。逐个判断是否满足**全部**约束。
只有每条约束都能被标题/详情/主图证据确认时,才算 satisfied=true。

只输出 JSON(items 顺序与给出的商品一致):
{{"items":[{{"idx":0,"satisfied":true}}]}}"""


def _extract_json(text: str) -> dict[str, Any]:
    t = s(text)
    if not t:
        return {}


    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1]
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(t[a : b + 1])
        except Exception:
            return {}
    return {}


def self_verify(content: list[dict[str, Any]], *, base_url: str, api_key: str, model: str, timeout: int = 120, retries: int = 6, no_think: bool = False, temperature: float | None = 0.0) -> str | None:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": content}],
    }


    if temperature is not None:
        body["temperature"] = temperature
    if no_think:

        body["extend_fields"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(retries):
        try:
            r = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", "")
            logger.warning("self_verify attempt=%d/%d failed: HTTP=%s %s", attempt + 1, retries, code, str(exc)[:150])
            if attempt < retries - 1:
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    return None


INFER_SYS = "你是购物需求分析器,从用户多轮对话中提炼核心意图与目标商品的硬约束。"
INFER_INSTRUCTION = """下面是用户在购物助手里的多轮输入(可能含图片)。请提炼:
1) intent:用户的核心购物意图(一句话);
2) constraints:目标商品必须满足的硬约束列表,每条为一个明确可判定的属性/条件(类目、品牌、型号、颜色、规格、"与图片同款"等)。
只输出 JSON:{"intent":"...","constraints":["...","..."]}"""


def infer_constraints(query_text: Any, query_images: Any, *, base_url: str, api_key: str, model: str,
                      timeout: int = 120, retries: int = 4, no_think: bool = False,
                      temperature: float | None = 0.0) -> tuple[str, str]:
    

    turns: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    def _touch(rnd: Any) -> dict[str, Any]:
        if rnd not in turns:
            turns[rnd] = {"text": None, "imgs": []}
            order.append(rnd)
        return turns[rnd]
    if isinstance(query_text, list):
        for i, it in enumerate(query_text):
            v = s(it.get("value")) if isinstance(it, dict) else s(it)
            rnd = it.get("chat_round", i) if isinstance(it, dict) else i
            if v:
                _touch(rnd)["text"] = v
    elif query_text:
        _touch(0)["text"] = s(query_text)
    if isinstance(query_images, list):
        for i, it in enumerate(query_images):
            u = s(it.get("value")) if isinstance(it, dict) else s(it)
            rnd = it.get("chat_round", i) if isinstance(it, dict) else i
            if u:
                _touch(rnd)["imgs"].append(u)
    try:
        order = sorted(order, key=lambda x: (x is None, x))
    except TypeError:
        pass
    content: list[dict[str, Any]] = [{"type": "text", "text": INFER_INSTRUCTION + "\n\n── 用户对话(按轮次顺序,文本+图) ──"}]
    for rnd in order:
        t = turns[rnd]
        if t["text"] is None and not t["imgs"]:
            continue
        content.append({"type": "text", "text": f"\n【第{rnd}轮】用户: {t['text'] or '(无文字,仅图片)'}"})
        for u in t["imgs"]:
            content.append({"type": "image_url", "image_url": {"url": u}})
    if len(content) == 1:
        content.append({"type": "text", "text": "\n(无文字)"})
    body: dict[str, Any] = {"model": model, "messages": [{"role": "system", "content": INFER_SYS}, {"role": "user", "content": content}]}
    if temperature is not None:
        body["temperature"] = temperature
    if no_think:
        body["extend_fields"] = {"chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(retries):
        try:
            r = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            obj = _extract_json(r.json()["choices"][0]["message"]["content"] or "")
            intent = s(obj.get("intent")) if isinstance(obj, dict) else ""
            cons = obj.get("constraints") if isinstance(obj, dict) else None
            attrs = "/".join(s(c) for c in cons if s(c)) if isinstance(cons, list) else s(cons)
            return intent, attrs
        except Exception as exc:
            logger.warning("infer_constraints attempt=%d/%d failed: %s", attempt + 1, retries, str(exc)[:120])
            if attempt < retries - 1:
                time.sleep(min(20.0, 2.0 * (2 ** attempt)))
    return "", ""


def _build_content(intent: str, attrs: str, ids: list[str], detail: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": INSTRUCTION.format(attrs=attrs, intent=intent or "(无)")}]
    for idx, iid in enumerate(ids):
        rec = detail.get(iid, {})
        block = f"商品[{idx}] 标题:{s(rec.get('title'))}"
        ev = evidence_text_for(rec)
        if ev:
            block += f"\n文本详情:{ev}"
        content.append({"type": "text", "text": block})
        pic = main_image_for(rec)
        if pic:
            content.append({"type": "image_url", "image_url": {"url": pic}})
    return content


def egvs_one(row: dict[str, Any], *, detail: dict[str, dict[str, Any]], base_url: str, api_key: str, model: str, topk: int, batch_size: int = 0, no_think: bool = False, temperature: float | None = 0.0, select_mode: str = "replace") -> dict[str, Any]:
    cid = s(row.get("conversation_id"))
    union_ids = uniq([s(x) for x in row.get("union_ids") or []])
    final_ids = [s(x) for x in row.get("final_ids") or []]
    sat_truth = set(s(x) for x in row.get("satisfied_ids") or [])


    gt_ids = set(s(x) for x in (row.get("gt_ids") or []))

    judged = [i for i in union_ids if i in detail]

    res = {
        "conversation_id": cid,
        "axis_a": s(row.get("axis_a")), "axis_b": s(row.get("axis_b")), "axis_c": s(row.get("axis_c")),
        "baseline_final_any": int(row.get("final_satisfied_any") or 0),
        "pool_any": int(row.get("pool_satisfied_any") or 0),
        "egvs_final_any": 0, "n_selected": 0, "selected_ids": [], "fallback": 0, "err": "",

        "has_gt": 1 if gt_ids else 0,
        "pool_id_hit": 1 if (gt_ids & set(union_ids)) else 0,
        "baseline_id_hit": 1 if (gt_ids & set(final_ids)) else 0,
        "egvs_id_hit": 0,
    }
    intent = s(row.get("user_intent_text")) or s(row.get("intent"))
    attrs = s(row.get("mandatory_attributes"))
    if not judged:
        res["err"] = "no_evidence"
    else:




        step = batch_size if batch_size and batch_size > 0 else len(judged)
        selected: list[str] = []
        any_ok = False
        for start in range(0, len(judged), max(1, step)):
            chunk = judged[start : start + max(1, step)]
            content = _build_content(intent, attrs, chunk, detail)
            obj = _extract_json(self_verify(content, base_url=base_url, api_key=api_key, model=model, no_think=no_think, temperature=temperature) or "")
            items = obj.get("items") if isinstance(obj, dict) else None
            if isinstance(items, list) and items:
                any_ok = True
                for it in items:
                    if isinstance(it, dict) and isinstance(it.get("idx"), int) and bool(it.get("satisfied")):
                        j = it["idx"]
                        if 0 <= j < len(chunk):
                            selected.append(chunk[j])
        if not any_ok:
            res["err"] = "verify_parse_fail"

        sat_list = uniq(selected)
        if select_mode == "augment":


            satset = set(sat_list); finset = set(final_ids)
            p1 = [i for i in final_ids if i in satset]
            p2 = [i for i in sat_list if i not in finset]
            p3 = [i for i in final_ids if i not in satset]
            selected = uniq(p1 + p2 + p3)[:topk]
            if not sat_list:
                res["fallback"] = 1
        else:
            selected = sat_list[:topk]
            if not selected:
                selected = final_ids[:topk]
                res["fallback"] = 1
        res["n_selected"] = len(selected)
        res["selected_ids"] = selected
        res["egvs_final_any"] = 1 if any(i in sat_truth for i in selected) else 0
        res["egvs_id_hit"] = 1 if (gt_ids & set(selected)) else 0
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-verify", required=True, help="pool_verify_report 输出的 pool_verify.jsonl")
    ap.add_argument("--offline-detail", required=True, help="离线商品详情 v0.6 jsonl")
    ap.add_argument("--mandatory-from", default="", help="可选:含 conversation_id/Mandatory_attributes/user_inent_text 的 json,补约束/意图")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--verify-model", default="")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--verify-batch-size", type=int, default=0, help="每次 self-verify 请求最多放几个候选(含图);0=全部放一个请求(原行为)。端点扛不住多图 payload 时设小,如 5,分批请求后合并满足项。")
    ap.add_argument("--verify-no-think", action="store_true", help="关闭 self-verify 的思考模式(Qwen 系 -think 模型:发 extend_fields.chat_template_kwargs.enable_thinking=False),大幅提速。")
    ap.add_argument("--verify-temperature", default="0.0", help="self-verify 的 temperature;传 none/omit/空 则不发该字段(gpt-5.5 / claude-opus 经 gateway 拒绝 temperature=0.0,须省略)。默认 0.0。")
    ap.add_argument("--select-mode", default="replace", choices=["replace", "augment"], help="replace(默认)=用满足项替换final,选不出兜底回final;augment=保留final只增删(满足项优先+未满足final填空),严格不劣、强模型egvs-id由负转正。")
    ap.add_argument("--infer-constraints", action="store_true", help="不用标注 Mandatory_attributes/user_intent,改用验证器 LLM 从原始对话(--mandatory-from 的 query_text/query_pic_img)推断约束与意图(公平/可部署口径,消除标注泄漏)。")
    ap.add_argument("--only-gap", action="store_true", help="只跑 pool=1,final=0 的会话(测方法救回率)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _t = s(args.verify_temperature).strip().lower()
    verify_temperature: float | None = None if _t in ("none", "omit", "") else float(_t)

    base_url = args.base_url or ""
    api_key = args.api_key or ""
    model = args.verify_model or ""
    if not (base_url and api_key and model):
        print("需要 --base-url/--api-key/--verify-model(self-verify 端点)", file=sys.stderr)
        return 2
    print(f"self-verify model = {model}", file=sys.stderr)

    rows = read_jsonl(args.pool_verify)

    if args.mandatory_from:
        d = json.loads(Path(args.mandatory_from).read_text(encoding="utf-8"))
        src = d if isinstance(d, list) else d.get("data", d)
        by = {s(r.get("conversation_id")): r for r in src}
        for r in rows:
            b = by.get(s(r.get("conversation_id")), {})
            r.setdefault("mandatory_attributes", s(b.get("Mandatory_attributes")))
            r.setdefault("user_intent_text", s(b.get("user_inent_text")))
            r.setdefault("gt_ids", split_ids(b.get("Manual_annotation_item_id")))
            r["_query_text"] = b.get("query_text")
            r["_query_images"] = b.get("query_pic_img")
    if args.only_gap:
        rows = [r for r in rows if int(r.get("pool_satisfied_any") or 0) == 1 and int(r.get("final_satisfied_any") or 0) == 0]

    if args.infer_constraints:
        print(f"infer-constraints: 用验证器({model})从原始对话推断约束/意图(替代标注),并发={args.concurrency} ...", file=sys.stderr)
        _iprog = {"n": 0}
        _ilock = threading.Lock()
        def _infer(r: dict[str, Any]) -> None:
            it, at = infer_constraints(r.get("_query_text"), r.get("_query_images"), base_url=base_url, api_key=api_key,
                                       model=model, no_think=args.verify_no_think, temperature=verify_temperature)
            r["mandatory_attributes"] = at
            r["user_intent_text"] = it
            r["_inferred_constraints"] = 1
            with _ilock:
                _iprog["n"] += 1
                if _iprog["n"] % 20 == 0:
                    print(f"  inferred {_iprog['n']}/{len(rows)}", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            list(ex.map(_infer, rows))
        print(f"infer-constraints done: {sum(1 for r in rows if s(r.get('mandatory_attributes')))}/{len(rows)} 条得到非空约束", file=sys.stderr)

    detail = load_offline_detail(args.offline_detail)
    print(f"convs={len(rows)}  offline_items={len(detail)}  topk={args.topk}", file=sys.stderr)

    prog = {"done": 0, "fb": 0, "pf": 0, "ne": 0}
    plock = threading.Lock()
    total = len(rows)

    def _one(r: dict[str, Any]) -> dict[str, Any]:
        res = egvs_one(r, detail=detail, base_url=base_url, api_key=api_key, model=model, topk=args.topk, batch_size=args.verify_batch_size, no_think=args.verify_no_think, temperature=verify_temperature, select_mode=args.select_mode)
        with plock:
            prog["done"] += 1
            prog["fb"] += int(res.get("fallback") or 0)
            if res.get("err") == "verify_parse_fail":
                prog["pf"] += 1
            elif res.get("err") == "no_evidence":
                prog["ne"] += 1
            if prog["done"] % 20 == 0 or prog["done"] == total:
                logger.info("progress %d/%d  fallback=%d parse_fail=%d no_evidence=%d",
                            prog["done"], total, prog["fb"], prog["pf"], prog["ne"])
        return res

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(_one, rows))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if not r["err"] or r["err"] == "verify_parse_fail"]
    n = len(ok)
    errs = Counter(r["err"] for r in results if r["err"])
    fb = sum(r["fallback"] for r in ok)
    print(f"\n=== EGVS self-verify (N={n}/{len(results)}, fallback={fb}) ===", file=sys.stderr)
    if n:
        print(f"  baseline final@any : {_rate(ok,'baseline_final_any'):.1%}", file=sys.stderr)
        print(f"  EGVS     final@any : {_rate(ok,'egvs_final_any'):.1%}   (Δ {_rate(ok,'egvs_final_any')-_rate(ok,'baseline_final_any'):+.1%})", file=sys.stderr)
        print(f"  oracle pool@any    : {_rate(ok,'pool_any'):.1%}   (EGVS 距上界 {_rate(ok,'pool_any')-_rate(ok,'egvs_final_any'):.1%})", file=sys.stderr)
    if errs:
        print(f"  errors: {dict(errs)}", file=sys.stderr)




    gt_rows = [r for r in ok if r.get("has_gt")]
    if gt_rows:
        bi, ei, pi = _rate(gt_rows, "baseline_id_hit"), _rate(gt_rows, "egvs_id_hit"), _rate(gt_rows, "pool_id_hit")
        print(f"\n=== id 匹配维度 (GT=Manual_annotation_item_id, N={len(gt_rows)}/{len(ok)} 有GT) ===", file=sys.stderr)
        print(f"  baseline final id-hit : {bi:.1%}", file=sys.stderr)
        print(f"  EGVS     final id-hit : {ei:.1%}   (Δ {ei-bi:+.1%})", file=sys.stderr)
        print(f"  oracle   pool id-hit  : {pi:.1%}   (EGVS 距上界 {pi-ei:.1%})", file=sys.stderr)
    else:
        print("\n=== id 匹配维度: 跳过(未提供 --mandatory-from 或无 GT) ===", file=sys.stderr)

    for axis in AXES:
        print(f"\n=== EGVS[judge] 按 {axis} ===", file=sys.stderr)
        by: dict[str, list[dict[str, Any]]] = {}
        for r in ok:
            by.setdefault(r[axis] or "NA", []).append(r)
        for lab in AXIS_ORDER[axis] + ("NA",):
            g = by.get(lab, [])
            if g:
                br, er, pr = _rate(g, "baseline_final_any"), _rate(g, "egvs_final_any"), _rate(g, "pool_any")
                print(f"  {lab:<15} n={len(g):<4} base={br:.1%}  EGVS={er:.1%} ({er-br:+.1%})  oracle={pr:.1%}", file=sys.stderr)

    if gt_rows:
        for axis in AXES:
            print(f"\n=== EGVS[id匹配] 按 {axis} ===", file=sys.stderr)
            by_id: dict[str, list[dict[str, Any]]] = {}
            for r in gt_rows:
                by_id.setdefault(r[axis] or "NA", []).append(r)
            for lab in AXIS_ORDER[axis] + ("NA",):
                g = by_id.get(lab, [])
                if g:
                    br, er, pr = _rate(g, "baseline_id_hit"), _rate(g, "egvs_id_hit"), _rate(g, "pool_id_hit")
                    print(f"  {lab:<15} n={len(g):<4} base={br:.1%}  EGVS={er:.1%} ({er-br:+.1%})  oracle={pr:.1%}", file=sys.stderr)

    print(f"\nwritten: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
