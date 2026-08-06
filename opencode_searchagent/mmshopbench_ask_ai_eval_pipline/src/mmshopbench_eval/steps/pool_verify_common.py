

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CARD_PAT = re.compile(r"<Card>(.*?)</Card>", re.S)
ID_PAT = re.compile(r'"?(?:item_id|id)"?\s*[:：]\s*"?(\d{6,19})')
SEARCH_TOOLS = {"platform_product_search", "product_image_search"}


def s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def uniq(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in seq if x and not (x in seen or seen.add(x))]


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []



    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out




def final_card_from_pred(pred: str) -> tuple[list[str], int]:
    
    for raw in CARD_PAT.findall(pred or ""):
        try:
            o = json.loads(raw)
        except Exception:
            continue
        if not isinstance(o, dict) or o.get("call_name") != "ItemListCard":
            continue
        p = o.get("input_parameters") if isinstance(o.get("input_parameters"), dict) else {}
        ids = [s(i) for i in (p.get("item_ids") or []) if s(i)]
        try:
            img_idx = int(p.get("img_idx", 0))
        except Exception:
            img_idx = 0
        return uniq(ids), img_idx
    return [], 0


def make_pred_with_ids(item_ids: list[str], img_idx: int = 0) -> str:
    
    card = {
        "call_name": "ItemListCard",
        "card_id": 0,
        "input_parameters": {"img_idx": img_idx, "item_ids": list(item_ids)},
    }
    return f"<Card>{json.dumps(card, ensure_ascii=False)}</Card>"




def _tool_name_map(steps: list[dict[str, Any]]) -> dict[str, str]:
    
    out: dict[str, str] = {}
    for st in steps:
        for msg in (st.get("output", {}) or {}).get("messages", []) or []:
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tcid = s(tc.get("id"))
                name = s(tc.get("name")) or s((tc.get("function") or {}).get("name"))
                if tcid and name:
                    out[tcid] = name
    return out


def _ids_from_tool_content(content: Any) -> list[str]:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    ids: list[str] = []

    try:
        obj = json.loads(text)
        items = obj.get("items") if isinstance(obj, dict) else None
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    iid = s(it.get("id") or it.get("item_id"))
                    if iid:
                        ids.append(iid)
    except Exception:
        pass
    if not ids:
        ids = ID_PAT.findall(text)
    return ids


def pool_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    
    name_by_id = _tool_name_map(steps)
    have_names = bool(name_by_id)
    pool: list[str] = []
    seen_tool_msgs: set[str] = set()
    for st in steps:
        for msg in (st.get("input", {}) or {}).get("messages", []) or []:
            if msg.get("role") != "tool":
                continue
            tcid = s(msg.get("tool_call_id"))
            if tcid and tcid in seen_tool_msgs:
                continue
            if tcid:
                seen_tool_msgs.add(tcid)
            name = name_by_id.get(tcid, "")

            if have_names and name and name not in SEARCH_TOOLS:
                continue
            pool.extend(_ids_from_tool_content(msg.get("content")))
    return uniq(pool)


def pool_by_run_id(trace_path: str) -> dict[str, list[str]]:
    
    out: dict[str, list[str]] = {}
    for rec in read_jsonl(trace_path):
        rid = s(rec.get("run_id"))
        if rid:
            out[rid] = pool_from_steps(rec.get("steps") or [])
    return out




def load_offline_detail(path: str) -> dict[str, dict[str, Any]]:
    
    out: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(path):
        iid = s(rec.get("item_id"))
        if iid:
            out[iid] = rec
    return out


def evidence_text_for(rec: dict[str, Any], maxlen: int = 700) -> str:
    
    if not rec:
        return ""
    parts = [s(rec.get("title"))]
    attrs = rec.get("attributes") if isinstance(rec.get("attributes"), dict) else {}
    for k in ("item_pv_edit", "item_ocr_info", "item_brand_name", "treecate_full_path"):
        v = s(attrs.get(k))
        if v:
            parts.append(f"{k}:{v}")
    dt = s(rec.get("detail_text"))
    if dt:
        parts.append(dt)
    return "\n".join(p for p in parts if p)[:maxlen]


def main_image_for(rec: dict[str, Any]) -> str:
    if not rec:
        return ""
    u = s(rec.get("main_image_url"))
    if u:
        return u
    imgs = rec.get("image_urls")
    return s(imgs[0]) if isinstance(imgs, list) and imgs else ""


def split_ids_field(text: str) -> list[str]:
    
    return uniq([t for t in re.split(r"[,，、\s]+", s(text)) if t])
