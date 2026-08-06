#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from mmshopbench_eval.config import load_env
from mmshopbench_eval.multiturn_benchmark import (
    first_nonempty,
    first_present,
    normalize_yes_no,
    normalize_image_urls,
    parse_float,
    parse_model_output,
    read_csv_records,
    rebuild_blocks_for_turn,
    safe_json_dumps,
    safe_json_loads,
    strip_card_protocol,
    truncate_text,
)
from mmshopbench_eval.partition import build_select_sql, normalize_partition_spec, partition_value

load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_TABLE = "your_dw_project.mmshopbench_multiturn_eval_benchmark"
DEFAULT_OUTPUT_TABLE = "your_dw_project.mmshopbench_multiturn_find_item_judge"

_DAILY_PREVIEW_CACHE: dict[tuple[str, str], Any] = {}


_CARD_CONVERT_MODULE_CACHE: dict[str, Any] = {}
_CARD_CONVERT_SCRIPT_NAME = "online_card_render.py"


_ITEM_PV_CACHE: dict[str, dict[str, str]] = {}
_ITEM_PV_LOCK = threading.Lock()
_ITEM_PV_URL = "https://product-api.example.com/pre/recommend"


_OFFLINE_MANIFEST_PATH = (
    "/path/to/opencode_searchagent/"
    "mmshopbench_ask_ai_benchmark/indexes/image/image_manifest_details.v0.5.unique.jsonl"
)

_MANIFEST_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_MANIFEST_LOCK = threading.Lock()


_OFFLINE_INFO_FIELDS = ("item_pv_edit", "item_ocr_info", "item_brand_name")

JUDGE_COLUMNS = [
    "sample_id",
    "session_id",
    "target_turn_id",
    "turn_index",
    "request_time",
    "chat_round",
    "multi_turn_type",
    "context_dependency",
    "expected_context",
    "user_intent_text",
    "mandatory_attributes",
    "候选商品id",
    "是否有满足要求的商品",
    "满足要求的商品id",
    "判断理由",
    "置信度",
    "judge_success",
    "judge_model",
    "judge_error",
    "raw_output",
    "judge_prompt_messages_json",
    "judge_input_payload_json",
    "judge_time",
]


JUDGE_SYSTEM_PROMPT = """你是一位识图购物 AI 购物助手的“找同款/找商品”命中评测官。

你的唯一任务：判断【当前轮 ItemListCard 里返回的商品】中，是否存在能够满足【用户输入（历史各轮 + 当前轮的图片和文字）】要求的商品。候选商品卡只有一种来源：ItemListCard(文搜，AI 依据文字/上下文筛出的商品)。

输入说明（已尽量还原成用户真实可见信息）：
- 会话按轮次给出。历史轮【只提供用户输入的图片和文字】，不包含 AI 的任何文本回复或商品卡——历史轮仅用于累积用户的需求、主体与偏好。
- 当前轮（目标轮）除了用户输入的图片和文字，还会给出 ItemListCard(文搜) 中的商品完整信息：标题、价格、店铺，以及真实补全信息——主图OCR文案(item_ocr_info)、结构化商品属性(item_pv_edit，如品牌/功率/材质/尺寸/颜色/款式/功能)、品牌(item_brand_name)，并附带商品主图。
- expected_context 是当前轮必须继承的历史主体、约束或用户偏好（可能为空）。
- 输入中【不包含】AI 的文本回复、引导词，也不要去揣测或依赖它们。只依据用户输入和商品本身信息判断。

【人工标注金标准】（本任务最高优先级，务必严格遵守）：
- user_intent_text：人工标注的“用户真正想找的商品”，是判断相关性的权威描述。
- mandatory_attributes：人工标注的“商品必须满足的属性”，是【硬性门槛】。一个商品只要缺失或违背其中【任意一条】必须属性，就【绝对不算命中】，无论它在其它方面多契合。
  （必须属性可能用“/”“、”等分隔多条，请逐条核对；每一条都必须能从商品的 item_pv_edit/item_ocr_info/标题/商品图中得到明确印证。）

判断步骤：
1) 先合并出一份“约束清单”，其中【人工标注】部分为硬性、优先级最高：
   a. 【硬性】mandatory_attributes 里的每一条必须属性——全部都要满足；
   b. 【硬性】user_intent_text 描述的目标商品——商品的品类/主体必须与之一致；
   c. 当前轮用户图片里的【隐含约束】——仔细看图，提取品牌/型号、颜色、材质、款式/形态、图案、核心主体（人/物/局部区域）、使用场景等视觉信息（用户往往不会用文字重复）；
   d. 当前轮 + 历史各轮用户文字里的【显式约束】——价格、规格、功能、数量、品牌、偏好等；
   e. expected_context 中要求必须继承的约束。
2) 逐个检查 ItemListCard(文搜) 里的商品：用该商品的 item_pv_edit（品牌/材质/颜色/尺寸/款式/功能等）、item_ocr_info、item_brand_name、标题、商品图，与约束清单【逐条比对】。
3) 一个商品被判定为“命中”，必须【同时】满足：
   (i) mandatory_attributes 的【全部】必须属性；
   (ii) 与 user_intent_text 描述的目标商品一致；
   (iii) 你从用户图片+文字（含历史轮）分析出的核心需求。
   只要有【至少一个】商品同时满足以上三点，就判定“是否有满足要求的商品=是”，并给出所有这样的商品 id；否则为“否”。

判定原则：
- 人工标注优先：mandatory_attributes 是不可逾越的硬门槛，任一必须属性不满足→该商品直接判为不命中；user_intent_text 与商品品类/主体不一致→不命中。
- 关键约束不符即视为不满足：如品牌不符、材质/颜色/规格/功能不符、认错核心主体、把 A 类目当成 B 类目、无视图片里的隐藏约束。
- 证据不足不等于满足：若商品缺少可核对的属性/标题/图片，无法确认它满足某条必须属性或核心需求，则不应把它算作满足。
- 除人工标注的必须属性外，次要属性不完全匹配但必须属性全中、核心主体/类目/关键规格都对，可算作命中。

输出必须是严格 JSON，不要 Markdown，不要额外解释。字段必须完整：
{
  "是否有满足要求的商品": "是/否",
  "满足要求的商品id": "满足要求的商品 id，多个用逗号分隔；没有则为空",
  "判断理由": "一句话说明判断依据，指出命中的商品命中了哪些约束，或为何全部不满足",
  "置信度": 0.0
}"""


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: str) -> list[dict[str, Any]]:
    
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("rows", "data", "records", "items"):
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            rows = [data]
    else:
        raise ValueError(f"不支持的 JSON 顶层结构: {type(data).__name__}")
    return [row for row in rows if isinstance(row, dict)]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(safe_json_dumps(row) + "\n")


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    extra = [key for row in rows for key in row.keys() if key not in JUDGE_COLUMNS]
    columns = list(dict.fromkeys(JUDGE_COLUMNS + extra))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_latest_response(row: dict[str, Any]) -> str:
    target = safe_json_loads(row.get("source_target_turn_json")) or {}
    source_row = target.get("source_row") if isinstance(target, dict) else {}
    raw_log = source_row.get("raw_log") if isinstance(source_row, dict) else {}
    if isinstance(raw_log, dict) and raw_log.get("response"):
        return str(raw_log.get("response") or "").strip()
    return first_nonempty(

        row.get("pred_model_output"),
        row.get("response"),
        row.get("latest_response"),
        row.get("model_output"),
        row.get("output_text"),
        target.get("assistant_text") if isinstance(target, dict) else "",
        source_row.get("model_output") if isinstance(source_row, dict) else "",
    ).strip()


def extract_protocol_cards(response_text: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for match in re.finditer(r"<Card>\s*(\{.*?\})\s*</Card>", response_text or "", re.DOTALL):
        parsed = safe_json_loads(match.group(1))
        if not isinstance(parsed, dict):
            continue
        params = parsed.get("input_parameters") or {}
        if not isinstance(params, dict):
            params = {}
        cards.append(
            {
                "call_name": parsed.get("call_name", ""),
                "card_id": parsed.get("card_id", ""),
                "title": params.get("title", ""),
                "search": params.get("search", []),
                "questions": params.get("questions", []),
                "item_ids": params.get("item_ids", []),
                "img_idx": params.get("img_idx", ""),
                "region": params.get("region", ""),
            }
        )
    return cards


CARD_TYPE_LABELS = {
    "image_same_items_card": "同款卡 ImageSameItemsCard",
    "item_list_card": "同款/精选商品卡 ItemListCard",
    "item_feeds_card": "推荐卡 ItemFeedsCard",
    "multi_region_image_same_items_card": "多主体同款卡 MultiRegionImageSameItemsCard",
    "follow_up_questions_card": "引导词 FollowUpQuestionsWithInputCard",
}


def int_arg(args: argparse.Namespace, name: str, default: int) -> int:
    try:
        return int(getattr(args, name, default))
    except Exception:
        return default


def turn_sort_value(turn: dict[str, Any], default: int) -> tuple[int, float, str]:
    for key in ("turn_index", "chat_round", "request_time"):
        text = str(turn.get(key) or "").strip()
        if not text:
            continue
        try:
            return (0, float(text), text)
        except Exception:
            return (0, 0.0, text)
    return (1, float(default), str(default))


def target_turn_matches(turn: dict[str, Any], row: dict[str, Any]) -> bool:
    target_id = str(row.get("target_turn_id") or "").strip()
    if target_id and str(turn.get("turn_id") or "") == target_id:
        return True
    try:
        return int(float(turn.get("turn_index") or -1)) == int(float(row.get("turn_index") or -2))
    except Exception:
        return False


def user_text_from_turn(turn: dict[str, Any]) -> str:
    return first_nonempty(turn.get("user_text"), turn.get("input_query"), turn.get("query"), turn.get("text"))


def image_urls_from_turn(turn: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("image_urls", "images", "pics", "image_url", "input_image_url"):
        urls.extend(normalize_image_urls(turn.get(key)))
    return list(dict.fromkeys(url for url in urls if url))







def product_api_request(base_url: str, params: dict, timeout: int = 10) -> str:
    
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if "outputString" in data:
            return str(data["outputString"])
        if "itemInfo" in data:
            return str(data["itemInfo"])
        if "result" in data:
            return str(data["result"])
        if "data" in data:
            return str(data["data"])
        if "message" in data:
            return str(data["message"])

        logger.warning(f"API 返回数据格式异常: {list(data.keys())}")
        return "未找到相关信息"

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        return "未找到相关信息"


def get_product_data_via_api(item_ids: list[str], tool_call_id: str, field: list[str] = None) -> str:
    
    if not item_ids:
        return "未提供有效的 itemIds"

    params = {
        "appid": "00000",
        "code": "item_attr_completion",
        "input_charset": "UTF-8",
        "output_charset": "UTF-8",
        "include_details": "true",
        "remarks": "all",
        "itemIds": ",".join(str(i) for i in item_ids),
        "toolCallId": tool_call_id,
        "field": field,
    }
    return product_api_request(_ITEM_PV_URL, params)




_ITEM_COMPLETION_FIELDS = [
    "item_ocr_info",
    "item_pv_edit",
]
_ITEM_COMPLETION_LABELS = {
    "item_price": "购物平台价",
    "item_ocr_info": "主图OCR文案",
    "item_title": "购物平台标题",
    "item_sale": "销量",
    "item_seller_id": "卖家ID",
    "item_pv_edit": "商品属性",
    "item_shop_name": "店铺",
    "item_sku_info": "SKU名称",
    "item_brand_name": "品牌",
}


def _normalize_field_value(value: Any) -> str:
    
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, (dict, list)):
        return safe_json_dumps(value)
    return str(value).strip()


def _extract_sku_name(value: Any) -> str:
    
    if isinstance(value, str):
        parsed = safe_json_loads(value)
        value = parsed if isinstance(parsed, dict) else value
    if isinstance(value, dict):
        return str(value.get("sku_name") or "").strip()
    return ""


def _extract_field_value(field: str, value: Any) -> str:
    
    if field == "item_sku_info":
        return _extract_sku_name(value)
    return _normalize_field_value(value)


def item_completion_fields(args: argparse.Namespace) -> list[str]:
    
    raw = getattr(args, "item_pv_fields", None)
    if not raw:
        return list(_ITEM_COMPLETION_FIELDS)
    if isinstance(raw, str):
        raw = raw.split(",")
    fields = [str(f).strip() for f in raw if str(f).strip()]
    return fields or list(_ITEM_COMPLETION_FIELDS)


def _request_item_completions(
    item_ids: list[str], args: argparse.Namespace
) -> dict[str, dict[str, str]]:
    
    if not item_ids:
        return {}
    params = {
        "appid": str(getattr(args, "item_pv_appid", "00000") or "00000"),
        "code": str(getattr(args, "item_pv_code", "item_attr_completion") or "item_attr_completion"),
        "input_charset": "UTF-8",
        "output_charset": "UTF-8",
        "include_details": "true",
        "remarks": "all",
        "itemIds": ",".join(str(i) for i in item_ids),
        "toolCallId": str(getattr(args, "item_pv_tool_call_id", "judge_item_pv") or "judge_item_pv"),

        "field": item_completion_fields(args),
    }
    url = str(getattr(args, "item_pv_url", _ITEM_PV_URL) or _ITEM_PV_URL)
    timeout = int_arg(args, "item_pv_timeout", 10)
    fields = item_completion_fields(args)
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning("item_completion 拉取失败 ids=%s: %s", item_ids, exc)
        return {}

    result: dict[str, dict[str, str]] = {}
    item_complete = data.get("itemAttrs") if isinstance(data, dict) else None
    if isinstance(item_complete, dict):
        for iid, info in item_complete.items():
            if isinstance(info, dict):
                result[str(iid)] = {
                    f: _extract_field_value(f, info.get(f)) for f in fields
                }
    return result


def fetch_item_completions(
    item_ids: list[str], args: argparse.Namespace
) -> dict[str, dict[str, str]]:
    
    wanted = [str(i).strip() for i in item_ids if str(i or "").strip()]
    out: dict[str, dict[str, str]] = {}
    to_query: list[str] = []
    with _ITEM_PV_LOCK:
        for iid in wanted:
            if iid in _ITEM_PV_CACHE:
                out[iid] = _ITEM_PV_CACHE[iid]
            elif iid not in to_query:
                to_query.append(iid)
    if to_query:
        fetched = _request_item_completions(to_query, args)
        with _ITEM_PV_LOCK:
            for iid in to_query:
                value = fetched.get(iid, {})
                _ITEM_PV_CACHE[iid] = value
                out[iid] = value
    return out


def item_pv_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "fetch_item_pv", True))







def load_offline_manifest(path: str) -> dict[str, dict[str, Any]]:
    
    key = str(path or _OFFLINE_MANIFEST_PATH)
    with _MANIFEST_LOCK:
        cached = _MANIFEST_CACHE.get(key)
        if cached is not None:
            return cached
        mapping: dict[str, dict[str, Any]] = {}
        try:
            with open(key, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    iid = str(rec.get("item_id") or "").strip()
                    if iid:
                        mapping[iid] = rec
            logger.info("加载离线商品 manifest %s，商品数=%d", key, len(mapping))
        except Exception as exc:
            logger.warning("加载离线 manifest 失败 %s: %s，商品信息将补不出来", key, exc)
        _MANIFEST_CACHE[key] = mapping
        return mapping


def get_offline_manifest(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return load_offline_manifest(str(getattr(args, "offline_manifest_path", _OFFLINE_MANIFEST_PATH) or _OFFLINE_MANIFEST_PATH))


def _parse_manifest_info(record: dict[str, Any]) -> dict[str, str]:
    
    raw = record.get("raw") if isinstance(record, dict) else None
    info = raw.get("info") if isinstance(raw, dict) else None
    parsed: Any = info
    if isinstance(info, str):
        text = info.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, str] = {}
    for field in _OFFLINE_INFO_FIELDS:
        value = str(parsed.get(field) or "").strip()
        if value:
            out[field] = value
    return out


def _manifest_image_url(record: dict[str, Any]) -> str:
    
    image_url = str(record.get("main_image_url") or "").strip()
    if image_url:
        return image_url
    for url in record.get("image_urls") or []:
        if str(url or "").strip():
            return str(url).strip()
    return ""


def _manifest_price(record: dict[str, Any]) -> str:
    value = record.get("price")
    if value in (None, ""):
        return ""
    return str(value)


def product_summary(item: dict[str, Any], rank: int, manifest: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    
    iid = str(first_nonempty(item.get("id"), item.get("item_id"), item.get("itemId")) or "").strip()
    record = (manifest or {}).get(iid) or {}
    return {
        "rank": rank,
        "item_id": iid,
        "title": first_nonempty(record.get("title")),
        "price": _manifest_price(record),
        "shop_name": first_nonempty(record.get("shop_name")),
        "sales_text": "",
        "detail_url": "",
        "image_url": _manifest_image_url(record),
        "shopping_tips": "",

        "item_completion": _parse_manifest_info(record),
    }


def summarize_card_block(
    block: dict[str, Any], args: argparse.Namespace, is_target: bool = False
) -> dict[str, Any] | None:
    btype = str(block.get("type") or "")
    if btype == "follow_up_questions_card":
        questions = [str(q).strip() for q in (block.get("questions") or []) if str(q).strip()]
        return {
            "block_type": btype,
            "card_name": CARD_TYPE_LABELS.get(btype, btype),
            "questions": questions,
        }
    if "card" not in btype:
        return None
    items = block.get("items") or []
    if not isinstance(items, list):
        items = []
    max_items = max(1, int_arg(args, "max_card_items_for_judge", 3))


    manifest = get_offline_manifest(args)
    product_items = [
        product_summary(item, rank=index, manifest=manifest)
        for index, item in enumerate(items[:max_items], start=1)
        if isinstance(item, dict)
    ]

    cap = int_arg(args, "item_pv_max_chars", 600)
    if cap > 0:
        for it in product_items:
            comp = it.get("item_completion") or {}
            it["item_completion"] = {
                field: (value[:cap] if len(value) > cap else value)
                for field, value in comp.items()
            }
    return {
        "block_type": btype,
        "card_name": CARD_TYPE_LABELS.get(btype, btype),
        "display_title": first_nonempty(block.get("display_title"), block.get("title")),
        "expand_text": str(block.get("expand_text") or ""),
        "item_count": len(items),
        "shown_item_count": len(product_items),
        "items": product_items,
    }


def visible_output_from_blocks(
    blocks: list[dict[str, Any]], args: argparse.Namespace, is_target: bool = False
) -> dict[str, Any]:
    text_parts: list[str] = []
    product_cards: list[dict[str, Any]] = []
    followups: list[str] = []
    for block in blocks:
        btype = str(block.get("type") or "")
        if btype == "markdown":
            text = strip_card_protocol(parse_model_output(block.get("content")))
            if text:
                text_parts.append(text)
            continue
        summary = summarize_card_block(block, args, is_target=is_target)
        if not summary:
            continue
        if btype == "follow_up_questions_card":
            followups.extend(summary.get("questions") or [])
        else:
            product_cards.append(summary)
    return {
        "text": "\n\n".join(text_parts).strip(),
        "product_cards": product_cards,
        "followups": list(dict.fromkeys(followups)),
    }


def preview_card_mode_for_judge(args: argparse.Namespace) -> str:
    return str(getattr(args, "judge_card_mode", "offline") or "offline")


def make_preview_args_for_judge(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        preview_card_mode=preview_card_mode_for_judge(args),
        daily_preview_root=getattr(args, "daily_preview_root", ""),
        preview_cookie=getattr(args, "preview_cookie", ""),
        preview_cookie_file=getattr(args, "preview_cookie_file", ""),
        preview_max_card_items=int_arg(args, "max_card_items_for_judge", 3),
        image_url_col="query_pic_img",
        query_col="query_text",
        model_output_col="model_output",
        trace_id_col="trace_id",
    )


def get_daily_preview_for_judge(args: argparse.Namespace) -> Any | None:
    mode = preview_card_mode_for_judge(args)
    if mode == "offline":
        return None
    root = str(getattr(args, "daily_preview_root", "") or "")
    key = (mode, root)
    if key in _DAILY_PREVIEW_CACHE:
        return _DAILY_PREVIEW_CACHE[key]

    logger.warning("online preview module not bundled; falling back to offline cards")
    _DAILY_PREVIEW_CACHE[key] = None
    return None


def find_card_convert_script(args: argparse.Namespace) -> Path | None:
    
    candidates: list[str] = []
    explicit_root = str(getattr(args, "daily_preview_root", "") or "").strip()
    env_root = os.getenv("ONLINE_CARD_RENDER_ROOT", "").strip()
    for value in (explicit_root, env_root):
        if value:
            candidates.append(value)
    for root in candidates:
        script = Path(root).expanduser() / _CARD_CONVERT_SCRIPT_NAME
        if script.exists():
            return script
    return None


def get_card_convert_module(args: argparse.Namespace) -> Any | None:
    
    script = find_card_convert_script(args)
    if not script:
        logger.warning(
            "未找到 %s（请设置 --daily-preview-root 或 ONLINE_CARD_RENDER_ROOT 指向 online_card_render 目录），"
            "目标轮商卡将为空", _CARD_CONVERT_SCRIPT_NAME,
        )
        _CARD_CONVERT_MODULE_CACHE[""] = None
        return None
    key = str(script)
    if key in _CARD_CONVERT_MODULE_CACHE:
        return _CARD_CONVERT_MODULE_CACHE[key]
    root = script.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module: Any | None = None
    try:
        spec = importlib.util.spec_from_file_location("_card_convert_adapter", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法为 {script} 创建 import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info("使用目标轮卡片转换脚本: %s", script)
    except Exception as exc:
        logger.warning("加载卡片转换脚本失败 %s: %s，目标轮商卡将为空", script, exc)
        module = None
    _CARD_CONVERT_MODULE_CACHE[key] = module
    return module


def card_convert_request_headers(args: argparse.Namespace) -> dict[str, str] | None:
    
    cookie = str(getattr(args, "preview_cookie", "") or "").strip()
    cookie_file = str(getattr(args, "preview_cookie_file", "") or "").strip()
    if not cookie and cookie_file:
        try:
            cookie = Path(cookie_file).expanduser().read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("读取 preview cookie 文件失败 %s: %s", cookie_file, exc)
    if not cookie:
        cookie = str(os.getenv("PREVIEW_COOKIE", "") or "").strip()
    return {"Cookie": cookie} if cookie else None


def target_turn_blocks_from_assistant_text(
    turn: dict[str, Any],
    module: Any | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    
    source_row = turn.get("source_row") if isinstance(turn.get("source_row"), dict) else {}
    assistant_text = first_present(
        turn.get("assistant_text"),
        turn.get("model_output"),
        turn.get("response"),
        source_row.get("model_output") if isinstance(source_row, dict) else "",
    )
    if module is None:

        return []

    full_text = module.parse_model_output(assistant_text)
    if not str(full_text or "").strip():
        return []

    max_items = max(1, int_arg(args, "max_card_items_for_judge", 3))

    blocks: list[dict[str, Any]] = []
    seen_card_sigs: set[tuple[Any, ...]] = set()

    for seq, seg in enumerate(module.split_segments(full_text)):

        if seg.get("kind") == "text":
            continue

        call_name = str(seg.get("call_name") or "")
        if call_name != "ItemListCard":
            continue
        params = seg.get("params") if isinstance(seg.get("params"), dict) else {}

        item_ids = params.get("item_ids") or []
        if isinstance(item_ids, str):
            item_ids = [item_ids]
        item_ids = [str(i).strip() for i in item_ids if str(i).strip()]
        sig = ("item_list", tuple(item_ids))
        if item_ids and sig not in seen_card_sigs:
            seen_card_sigs.add(sig)

            items = [{"item_id": iid} for iid in item_ids[:max_items]]
            if items:
                blocks.append({
                    "seq": seq,
                    "type": "item_list_card",
                    "display_title": "",
                    "expand_text": "AI筛选的相关商品",
                    "first_show_count": 1,
                    "items": items,
                })

    return blocks


def rebuild_visible_blocks_for_judge(turn: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    offline_blocks = rebuild_blocks_for_turn(turn, turn.get("blocks"))

    get_daily_preview_for_judge(args)
    return offline_blocks


def fallback_visible_turns(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    latest_response = extract_latest_response(row)
    target = safe_json_loads(row.get("source_target_turn_json")) or {}
    source_row = target.get("source_row") if isinstance(target, dict) else {}
    if not isinstance(source_row, dict):
        source_row = {}
    output = visible_output_from_blocks(
        rebuild_blocks_for_turn(
            {
                "assistant_text": latest_response or row.get("model_output") or row.get("output_text"),
                "source_row": source_row,
            },
            [],
        ),
        args,

        is_target=True,
    )

    output = {
        "text": "",
        "product_cards": [
            card for card in (output.get("product_cards") or [])
            if str(card.get("block_type") or "") == "item_list_card"
        ],
        "followups": [],
    }
    return [
        {
            "turn_index": row.get("turn_index", ""),
            "is_target_turn": True,
            "user": {
                "text": str(row.get("current_user_input") or ""),
                "image_urls": [],
            },
            "assistant_visible_output": output,
        }
    ]







def is_num57_row(row: dict[str, Any]) -> bool:
    
    return isinstance(row.get("round_results"), list) and (
        "important_chat_round" in row or "pred_model_output" in row
    )


def _num57_round_map(entries: Any, value_key: str) -> dict[int, Any]:
    
    mapping: dict[int, Any] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            rnd = int(entry.get("chat_round"))
        except (TypeError, ValueError):
            continue
        mapping[rnd] = entry.get(value_key)
    return mapping


def _num57_important_round(row: dict[str, Any]) -> int:
    
    try:
        important = int(row.get("important_chat_round"))
    except (TypeError, ValueError):
        important = 0
    if important >= 1:
        return important
    rounds = [
        rnd
        for rnd in _num57_round_map(row.get("round_results"), "model_output").keys()
    ]
    return max(rounds) if rounds else 1


def _num57_images(value: Any) -> list[str]:
    
    if not value:
        return []
    if isinstance(value, list):
        return [url for url in value if url]
    return [value]


def build_visible_turns_from_num57(
    row: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    
    important = _num57_important_round(row)
    text_by_round = _num57_round_map(row.get("query_text"), "value")
    img_by_round = _num57_round_map(row.get("query_pic_img"), "value")
    pred_output = str(row.get("pred_model_output") or "")

    module = get_card_convert_module(args)

    visible_turns: list[dict[str, Any]] = []
    for rnd in range(1, important + 1):
        is_target = rnd == important
        images = _num57_images(img_by_round.get(rnd))
        text = str(text_by_round.get(rnd) or "").strip()

        if is_target:
            blocks = target_turn_blocks_from_assistant_text(
                {"assistant_text": pred_output, "turn_index": rnd},
                module,
                args,
            )
            output = visible_output_from_blocks(blocks, args, is_target=True)
        else:

            output = {}
        visible_turns.append(
            {
                "turn_index": rnd,
                "turn_id": "",
                "is_target_turn": is_target,
                "user": {"text": text, "image_urls": images},
                "assistant_visible_output": output,
            }
        )
    return visible_turns


def normalize_num57_row(row: dict[str, Any]) -> dict[str, Any]:
    
    if not is_num57_row(row):
        return row
    important = _num57_important_round(row)
    conversation_id = str(row.get("conversation_id") or "")
    if not row.get("sample_id"):
        row["sample_id"] = conversation_id
    row.setdefault("session_id", conversation_id)
    row.setdefault("turn_index", important)
    row.setdefault("chat_round", important)
    return row


def build_visible_turns(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if is_num57_row(row):
        return build_visible_turns_from_num57(row, args)
    session = safe_json_loads(row.get("source_session_json")) or {}
    turns = session.get("turns") if isinstance(session, dict) else []
    if not isinstance(turns, list) or not turns:
        return fallback_visible_turns(row, args)

    ordered = sorted(
        [turn for turn in turns if isinstance(turn, dict)],
        key=lambda item: turn_sort_value(item, default=0),
    )
    card_convert_module = get_card_convert_module(args)
    visible_turns: list[dict[str, Any]] = []
    appended_meta: list[tuple[int, dict[str, Any]]] = []
    for index, turn in enumerate(ordered, start=1):
        is_target = target_turn_matches(turn, row)
        try:
            turn_index = int(float(turn.get("turn_index") or index))
        except Exception:
            turn_index = index
        try:
            target_index = int(float(row.get("turn_index") or 0))
        except Exception:
            target_index = 0
        if not is_target and target_index and turn_index > target_index:
            continue

        if is_target:

            blocks = target_turn_blocks_from_assistant_text(turn, card_convert_module, args)
            output = visible_output_from_blocks(blocks, args, is_target=True)
        else:

            output = {}
        visible_turns.append(
            {
                "turn_index": turn_index,
                "turn_id": str(turn.get("turn_id") or ""),
                "is_target_turn": is_target,
                "user": {
                    "text": user_text_from_turn(turn),
                    "image_urls": image_urls_from_turn(turn),
                },
                "assistant_visible_output": output,
            }
        )
        appended_meta.append((index - 1, turn))
        if is_target:
            break

    if not any(turn.get("is_target_turn") for turn in visible_turns) and visible_turns:

        visible_turns[-1]["is_target_turn"] = True
        _last_pos, last_turn = appended_meta[-1]
        blocks = target_turn_blocks_from_assistant_text(last_turn, card_convert_module, args)
        visible_turns[-1]["assistant_visible_output"] = visible_output_from_blocks(blocks, args, is_target=True)
    return visible_turns or fallback_visible_turns(row, args)


def format_product_line(item: dict[str, Any]) -> str:
    
    pieces = [f"{item.get('rank')}. {item.get('title') or '未知商品'}"]
    if item.get("price"):
        pieces.append(f"价格: {item['price']}")
    if item.get("shop_name"):
        pieces.append(f"店铺: {item['shop_name']}")
    if item.get("sales_text"):
        pieces.append(f"销量/标签: {item['sales_text']}")
    if item.get("shopping_tips"):
        pieces.append(f"卖点: {item['shopping_tips']}")


    completion = item.get("item_completion") or {}
    for field, value in completion.items():
        value = str(value or "").strip()
        if value:
            label = _ITEM_COMPLETION_LABELS.get(field, field)
            pieces.append(f"{label}({field}): {value}")
    return " | ".join(str(piece) for piece in pieces if str(piece).strip())


_CARD_SOURCE_LABELS = {
    "item_list_card": "ItemListCard(文搜)",
}


def format_itemlist_cards_text(output: dict[str, Any]) -> str:
    
    lines: list[str] = []
    cards = output.get("product_cards") or []
    if not cards:
        return "【ItemListCard】\n（当前轮没有 ItemListCard 商品）"
    for card in cards:
        item_count = card.get("item_count", 0)
        shown = card.get("shown_item_count", len(card.get("items") or []))
        label = _CARD_SOURCE_LABELS.get(str(card.get("block_type") or ""), "候选商品卡")
        lines.append(f"【{label}】（共{item_count}件，展示前{shown}件）")
        items = card.get("items") or []
        if items:
            for item in items:
                iid = str(item.get("item_id") or "").strip()
                id_part = f"[item_id={iid}] " if iid else ""
                lines.append("  " + id_part + format_product_line(item))
        else:
            lines.append("  （没有可见商品详情）")
    return "\n".join(lines)


def _collect_target_item_ids(target_output: dict[str, Any]) -> list[str]:
    
    ids: list[str] = []
    for card in (target_output or {}).get("product_cards") or []:
        for item in card.get("items") or []:
            iid = str(item.get("item_id") or "").strip()
            if iid and iid not in ids:
                ids.append(iid)
    return ids


def _collect_structural_hints(payload: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    visible = payload.get("visible_conversation") or []
    total_turns = len(visible)
    target_idx = next((t.get("turn_index") for t in visible if t.get("is_target_turn")), "?")
    hints.append(f"会话共 {total_turns} 轮，目标轮为第 {target_idx} 轮")

    multi_turn_type = payload.get("multi_turn_type") or ""
    if multi_turn_type:
        hints.append(f"多轮类型：{multi_turn_type}")
    context_dep = payload.get("context_dependency") or ""
    if context_dep:
        hints.append(f"上下文依赖：{context_dep}")

    target_turn = next((t for t in visible if t.get("is_target_turn")), {})
    target_output = target_turn.get("assistant_visible_output") or {}
    item_ids = _collect_target_item_ids(target_output)
    if item_ids:
        hints.append(f"当前轮 ItemListCard(文搜) 共 {len(item_ids)} 个候选商品，item_id 列表：{', '.join(item_ids)}")
    else:
        hints.append("当前轮 ItemListCard 没有可判断的商品")
    return hints


def build_judge_user_content(payload: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []


    user_intent_text = str(payload.get("user_intent_text") or "").strip()
    mandatory_attributes = str(payload.get("mandatory_attributes") or "").strip()
    meta_lines = [
        f"sample_id: {payload.get('sample_id')}",
        f"multi_turn_type: {payload.get('multi_turn_type')}",
        f"context_dependency: {payload.get('context_dependency')}",
        f"expected_context: {payload.get('expected_context')}",
    ]
    gold_lines = [
        "★★★ 人工标注金标准（评分最高优先级，必须严格遵守）★★★",
        f"user_intent_text（人工标注-用户想找的商品）: {user_intent_text or '（空）'}",
        f"mandatory_attributes（人工标注-商品必须满足的属性，硬性门槛，逐条全中才算命中）: {mandatory_attributes or '（空）'}",
    ]
    content.append({
        "type": "text",
        "text": (
            "═══ 找商品命中判断任务 ═══\n"
            "下面给出多轮用户输入（历史轮只有用户图片+文字），以及当前轮 ItemListCard(文搜) 里的商品完整信息。\n"
            "请判断这些商品中是否有能【同时】满足“人工标注金标准”和“用户图文需求”的，只返回严格 JSON。\n"
            + "\n".join(meta_lines)
            + "\n\n"
            + "\n".join(gold_lines)
        ),
    })


    max_product_images = max(0, int_arg(args, "max_product_images_per_card", 2))
    for turn in payload.get("visible_conversation") or []:
        is_target = turn.get("is_target_turn")
        marker = "★ 当前轮（含待判断的 ItemListCard 商品）★" if is_target else "历史轮，仅用户输入作约束"
        user = turn.get("user") or {}

        content.append({"type": "text", "text": f"\n═══ 第{turn.get('turn_index')}轮（{marker}） ═══"})


        content.append({"type": "text", "text": "── 用户输入 ──"})
        image_urls = user.get("image_urls") or []
        if image_urls:
            for img_index, url in enumerate(image_urls, start=1):
                content.append({"type": "text", "text": f"用户图片{img_index}:"})
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
        text = str(user.get("text") or "").strip()
        content.append({"type": "text", "text": f"用户文字: {text if text else '（无文字）'}"})


        if not is_target:
            continue


        output = turn.get("assistant_visible_output") or {}
        content.append({"type": "text", "text": "── 当前轮 ItemListCard 商品 ──\n" + format_itemlist_cards_text(output)})


        if is_target:
            for card in output.get("product_cards") or []:
                images_added = 0
                for item in card.get("items") or []:
                    image_url = str(item.get("image_url") or "").strip()
                    if not image_url:
                        continue
                    if images_added >= max_product_images:
                        break
                    images_added += 1
                    content.append({
                        "type": "text",
                        "text": f"商品图: {card.get('card_name')} 第{item.get('rank')}个商品 {item.get('title') or ''} {item.get('price') or ''}",
                    })
                    content.append({"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}})


    hints = _collect_structural_hints(payload)
    content.append({"type": "text", "text": "\n═══ 结构信息 ═══\n" + "\n".join(hints)})


    content.append({
        "type": "text",
        "text": (
            "\n═══ 输出要求 ═══\n"
            "第一步：把 mandatory_attributes 拆成逐条必须属性；结合 user_intent_text，再从当前轮用户图片提取隐藏约束、"
            "合并当前轮+历史各轮用户文字约束（及 expected_context），形成完整约束清单。\n"
            "第二步：用当前轮每个 ItemListCard(文搜) 商品的 item_pv_edit/item_ocr_info/item_brand_name/标题/商品图逐条核对。\n"
            "第三步：一个商品要算命中，必须【同时】满足——(i) mandatory_attributes 的全部必须属性；"
            "(ii) 与 user_intent_text 描述的目标商品一致；(iii) 用户图文分析出的核心需求。三者缺一不可。\n"
            "只要有至少一个商品同时满足以上三点，即“是否有满足要求的商品=是”，并在“满足要求的商品id”里列出所有命中的 item_id（用上面标注的 item_id）；否则为“否”。\n"
            "任一必须属性不满足、或与 user_intent_text 品类/主体不符、或证据不足无法确认，均不算命中。\n"
            "请按系统提示的字段定义完整输出 JSON。"
        ),
    })
    return content


def build_judge_payload(row: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or argparse.Namespace(max_card_items_for_judge=3, max_product_images_per_card=2)
    latest_response = extract_latest_response(row)
    visible_turns = build_visible_turns(row, args)
    target_turn = next((turn for turn in visible_turns if turn.get("is_target_turn")), visible_turns[-1] if visible_turns else {})
    target_output = target_turn.get("assistant_visible_output") if isinstance(target_turn, dict) else {}

    return {
        "sample_id": row.get("sample_id", ""),
        "session_id": row.get("session_id", ""),
        "target_turn_id": row.get("target_turn_id", ""),
        "turn_index": row.get("turn_index", ""),
        "request_time": row.get("request_time", ""),
        "chat_round": row.get("chat_round", ""),
        "multi_turn_type": row.get("multi_turn_type", ""),
        "context_dependency": row.get("context_dependency", ""),
        "expected_context": row.get("expected_context", ""),

        "user_intent_text": first_present(row.get("user_inent_text"), row.get("user_intent_text")),
        "mandatory_attributes": first_present(row.get("Mandatory_attributes"), row.get("mandatory_attributes")),
        "visible_conversation": visible_turns,
        "target_user_input": target_turn.get("user", {}) if isinstance(target_turn, dict) else {},

        "target_item_ids": _collect_target_item_ids(target_output if isinstance(target_output, dict) else {}),
        "latest_response_product_cards": (target_output or {}).get("product_cards", []) if isinstance(target_output, dict) else [],
        "latest_response_raw_protocol_cards": extract_protocol_cards(latest_response),
    }


def build_judge_messages(
    row: dict[str, Any],
    args: argparse.Namespace | None = None,
    *,
    payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payload = payload or build_judge_payload(row, args)
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_user_content(payload, args or argparse.Namespace(max_product_images_per_card=2))},
    ]


_REQUIRED_JUDGE_FIELDS = {"是否有满足要求的商品", "满足要求的商品id"}


def parse_judge_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    if not _REQUIRED_JUDGE_FIELDS.issubset(parsed):
        return None
    return parsed


def score_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return parse_float(value)


def normalize_result(
    parsed: dict[str, Any], row: dict[str, Any], candidate_item_ids: list[str] | None = None
) -> dict[str, Any]:
    confidence = score_float(parsed.get("置信度"))
    return {
        "sample_id": row.get("sample_id", ""),
        "session_id": row.get("session_id", ""),
        "target_turn_id": row.get("target_turn_id", ""),
        "turn_index": row.get("turn_index", ""),
        "request_time": row.get("request_time", ""),
        "chat_round": row.get("chat_round", ""),
        "multi_turn_type": row.get("multi_turn_type", ""),
        "context_dependency": row.get("context_dependency", ""),
        "expected_context": row.get("expected_context", ""),
        "user_intent_text": first_present(row.get("user_inent_text"), row.get("user_intent_text")),
        "mandatory_attributes": first_present(row.get("Mandatory_attributes"), row.get("mandatory_attributes")),
        "候选商品id": ",".join(candidate_item_ids or []),
        "是否有满足要求的商品": normalize_yes_no(parsed.get("是否有满足要求的商品")),
        "满足要求的商品id": str(parsed.get("满足要求的商品id") or ""),
        "判断理由": str(parsed.get("判断理由") or ""),
        "置信度": confidence if confidence is not None else "",
    }


def extract_response_text(resp_json: dict[str, Any]) -> str:
    data = resp_json.get("data") or resp_json
    choices = data.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or first.get("delta") or {}
    content = message.get("content") or first.get("text") or ""
    if isinstance(content, list):
        return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def judge_one(row: dict[str, Any], args: argparse.Namespace, session: requests.Session | None = None, rate_limiter: "_RateLimiter | None" = None) -> dict[str, Any]:
    payload = build_judge_payload(row, args)
    messages = build_judge_messages(row, args, payload=payload)
    body = {
        "platformInput": {"model": args.model},
        "messages": messages,
        "temperature": args.temperature,
        "toolChoice": "none",
    }
    headers = {"X-API-Key": args.api_key, "accept": "*/*", "Content-Type": "application/json"}
    candidate_item_ids = payload.get("target_item_ids") or []
    last_error = ""
    usage: dict[str, int] = {}
    for attempt in range(args.retries + 1):
        if rate_limiter:
            rate_limiter.wait()
        try:
            http_post = session.post if session else requests.post
            response = http_post(args.api_url, headers=headers, json=body, timeout=args.timeout)
            data = response.json()
            if data.get("success") is False:
                last_error = str(data.get("message") or "API error")
                logger.warning("Quality judge API error attempt=%d: %s", attempt + 1, last_error)
                time.sleep(args.retry_backoff_seconds * (attempt + 1))
                continue
            resp_data = data.get("data") or data
            if isinstance(resp_data, dict) and isinstance(resp_data.get("usage"), dict):
                usage = resp_data["usage"]
            raw_text = extract_response_text(data)
            parsed = parse_judge_json(raw_text)
            if parsed is not None:
                result = normalize_result(parsed, row, candidate_item_ids)
                result.update({"judge_success": "是", "judge_model": args.model, "judge_error": "", "raw_output": raw_text[:4000]})
                result["_usage"] = usage
                if args.save_prompt_messages:
                    result["judge_prompt_messages_json"] = safe_json_dumps(messages)
                    result["judge_input_payload_json"] = safe_json_dumps(payload)
                return result
            last_error = f"JSON parse failed: {raw_text[:200]}"
            logger.warning("Quality judge parse error attempt=%d: %s", attempt + 1, last_error)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Quality judge request error attempt=%d: %s", attempt + 1, last_error)
        time.sleep(args.retry_backoff_seconds * (attempt + 1))

    result = normalize_result({}, row, candidate_item_ids)
    result.update({"judge_success": "否", "judge_model": args.model, "judge_error": last_error, "raw_output": ""})
    result["_usage"] = usage
    if args.save_prompt_messages:
        result["judge_prompt_messages_json"] = safe_json_dumps(messages)
        result["judge_input_payload_json"] = safe_json_dumps(payload)
    return result


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_json:
        rows = read_json(args.input_json)
        rows = [normalize_num57_row(row) for row in rows]
        logger.info("Loaded %d records from JSON file %s", len(rows), args.input_json)
        return rows
    if args.input_jsonl:
        return read_jsonl(args.input_jsonl)
    if args.input_csv:
        return read_csv_records(args.input_csv)
    if not args.input_table:
        raise ValueError("需要 --input-json / --input-jsonl / --input-csv / --input-table 之一")
    from mmshopbench_eval.dw_handler import DWHandler

    handler = DWHandler(
        access_id=args.dw_access_id or None,
        access_key=args.dw_access_key or None,
        project=args.dw_project or None,
        endpoint=args.dw_endpoint or None,
    )
    sql = args.input_sql or build_select_sql(
        args.input_table,
        partition=args.input_partition,
        extra_where=args.input_where,
        limit=args.limit,
    )
    rows = handler.read_table_to_dict(sql)
    logger.info("Loaded %d benchmark rows", len(rows))
    return rows


class _RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_time = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._last_time + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_time = time.monotonic()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return done
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            sid = str(row.get("sample_id") or "")
            if sid:
                done[sid] = row
        except Exception:
            continue
    return done


def _append_checkpoint(checkpoint_path: Path, rows: list[dict[str, Any]]) -> None:
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(safe_json_dumps(row) + "\n")


def run_judge(rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path | None = None) -> list[dict[str, Any]]:
    checkpoint_path = (output_dir or Path(".")) / "judge_checkpoint.jsonl"
    resumed: dict[str, dict[str, Any]] = {}
    if args.resume and output_dir:
        resumed = _load_checkpoint(checkpoint_path)
        if resumed:
            logger.info("Resumed %d already-judged rows from checkpoint", len(resumed))

    pending = [(i, row) for i, row in enumerate(rows) if str(row.get("sample_id") or "") not in resumed]
    results: list[dict[str, Any] | None] = [None] * len(rows)
    for sid, result in resumed.items():
        for i, row in enumerate(rows):
            if str(row.get("sample_id") or "") == sid:
                results[i] = result
                break

    if not pending:
        logger.info("All %d rows already judged from checkpoint", len(rows))
        return [r for r in results if r is not None]

    http_session = requests.Session()
    http_session.headers.update({"X-API-Key": args.api_key, "accept": "*/*", "Content-Type": "application/json"})
    rate_limiter = _RateLimiter(args.request_delay_seconds) if args.request_delay_seconds > 0 else None
    checkpoint_buffer: list[dict[str, Any]] = []
    checkpoint_lock = threading.Lock()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _one(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return index, judge_one(row, args, session=http_session, rate_limiter=rate_limiter)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {executor.submit(_one, i, row): i for i, row in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    index, result = future.result()
                except Exception as exc:
                    index = futures[future]
                    row = rows[index]
                    logger.error("Judge row %d (sample_id=%s) crashed: %s", index, row.get("sample_id"), exc)
                    result = normalize_result({}, row)
                    result.update({"judge_success": "否", "judge_model": args.model, "judge_error": str(exc), "raw_output": ""})

                usage = result.pop("_usage", {})
                if isinstance(usage, dict):
                    for k in total_usage:
                        total_usage[k] += int(usage.get(k) or 0)

                results[index] = result
                if output_dir:
                    with checkpoint_lock:
                        checkpoint_buffer.append(result)
                        if len(checkpoint_buffer) >= 20:
                            _append_checkpoint(checkpoint_path, checkpoint_buffer)
                            checkpoint_buffer.clear()
                if done % 10 == 0 or done == len(pending):
                    logger.info("Judged %d/%d (total %d)", done, len(pending), len(rows))
    finally:
        http_session.close()
        if output_dir and checkpoint_buffer:
            _append_checkpoint(checkpoint_path, checkpoint_buffer)
            checkpoint_buffer.clear()

    final = [r for r in results if r is not None]
    for r in final:
        r.pop("_usage", None)
    if any(total_usage.values()):
        logger.info("Token usage — prompt: %d, completion: %d, total: %d", total_usage["prompt_tokens"], total_usage["completion_tokens"], total_usage["total_tokens"])
        for r in final:
            r["_total_usage"] = total_usage
    return final


def mean_score(rows: list[dict[str, Any]], field: str) -> float | None:
    nums = [value for value in (score_float(row.get(field)) for row in rows) if value is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if str(row.get("judge_success") or "") == "是"]
    total = len(rows)
    hit = sum(1 for row in success_rows if str(row.get("是否有满足要求的商品") or "") == "是")
    miss = sum(1 for row in success_rows if str(row.get("是否有满足要求的商品") or "") == "否")
    usage = {}
    for row in rows:
        u = row.pop("_total_usage", None)
        if isinstance(u, dict) and not usage:
            usage = u
    report = {
        "row_count": total,
        "judge_success_count": len(success_rows),
        "judge_success_rate": round(len(success_rows) / total, 4) if total else 0.0,
        "hit_count": hit,
        "miss_count": miss,
        "hit_rate": round(hit / len(success_rows), 4) if success_rows else 0.0,
        "avg_confidence": mean_score(success_rows, "置信度"),
        "hit_counts": dict(Counter(str(row.get("是否有满足要求的商品") or "") for row in success_rows)),
        "multi_turn_type_counts": dict(Counter(str(row.get("multi_turn_type") or "") for row in success_rows)),
        "context_dependency_counts": dict(Counter(str(row.get("context_dependency") or "") for row in success_rows)),
    }
    if usage:
        report["token_usage"] = usage
    return report


def ensure_output_table(handler: Any, table_name: str) -> None:
    schema = """
        sample_id STRING COMMENT 'target-turn 样本ID',
        session_id STRING COMMENT 'conversation/session ID',
        target_turn_id STRING COMMENT '目标轮ID',
        turn_index BIGINT COMMENT '目标轮序号',
        request_time STRING COMMENT '请求时间',
        chat_round STRING COMMENT 'chat_round',
        multi_turn_type STRING COMMENT '多轮类型',
        context_dependency STRING COMMENT '上下文依赖',
        expected_context STRING COMMENT '必须继承的上下文',
        user_intent_text STRING COMMENT '人工标注-用户想找的商品',
        mandatory_attributes STRING COMMENT '人工标注-商品必须满足的属性',
        candidate_item_ids STRING COMMENT '当前轮ItemListCard候选商品id',
        has_matching_item STRING COMMENT '是否有满足要求的商品',
        matching_item_ids STRING COMMENT '满足要求的商品id',
        judge_reason STRING COMMENT '判断理由',
        confidence DOUBLE COMMENT '置信度',
        judge_success STRING COMMENT 'Judge是否成功',
        judge_model STRING COMMENT 'Judge模型',
        judge_error STRING COMMENT 'Judge错误',
        raw_output STRING COMMENT 'Judge原始输出',
        judge_time STRING COMMENT '评测时间'
    """
    handler.create_table_if_not_exists(
        table_name=table_name,
        schema=schema,
        partition_spec="ds STRING COMMENT '日期分区'",
        lifecycle=365,
        comment="识图购物多轮 target-turn ItemListCard 找商品命中 LLM judge 结果",
    )


def dw_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "session_id": row.get("session_id", ""),
        "target_turn_id": row.get("target_turn_id", ""),
        "turn_index": int(float(row.get("turn_index") or 0)),
        "request_time": row.get("request_time", ""),
        "chat_round": row.get("chat_round", ""),
        "multi_turn_type": row.get("multi_turn_type", ""),
        "context_dependency": row.get("context_dependency", ""),
        "expected_context": row.get("expected_context", ""),
        "user_intent_text": row.get("user_intent_text", ""),
        "mandatory_attributes": row.get("mandatory_attributes", ""),
        "candidate_item_ids": row.get("候选商品id", ""),
        "has_matching_item": row.get("是否有满足要求的商品", ""),
        "matching_item_ids": row.get("满足要求的商品id", ""),
        "judge_reason": row.get("判断理由", ""),
        "confidence": score_float(row.get("置信度")),
        "judge_success": row.get("judge_success", ""),
        "judge_model": row.get("judge_model", ""),
        "judge_error": row.get("judge_error", ""),
        "raw_output": row.get("raw_output", ""),
        "judge_time": row.get("judge_time", ""),
    }


def write_to_dw(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.no_write_dw or not args.output_table:
        return
    from mmshopbench_eval.dw_handler import DWHandler

    handler = DWHandler(
        access_id=args.dw_access_id or None,
        access_key=args.dw_access_key or None,
        project=args.dw_project or None,
        endpoint=args.dw_endpoint or None,
    )
    ensure_output_table(handler, args.output_table)
    partition = normalize_partition_spec(args.output_partition)
    handler.write_table_from_dict(
        args.output_table,
        [dw_record(row) for row in rows],
        partition=partition,
        overwrite=args.output_overwrite,
    )
    logger.info("DW judge table written: %s/%s records=%d", args.output_table, partition, len(rows))


def write_dw_records_json(rows: list[dict[str, Any]], path: str, partition_ds: str) -> None:
    
    records = []
    for row in rows:
        record = dw_record(row)
        record["ds"] = partition_ds
        records.append(record)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("DW-format JSON written: %s records=%d", path, len(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多轮 target-turn 文本/商卡/引导词 LLM judge")
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE, help="输入 benchmark DW 表")
    parser.add_argument("--input-partition", default="", help="输入 benchmark 分区，如 ds=20260608_multiturn_v1_100_llm")
    parser.add_argument("--input-where", default="", help="输入表额外过滤条件")
    parser.add_argument("--input-sql", default="", help="自定义输入 SQL，优先级最高")
    parser.add_argument("--input-json", default="", help="本地 benchmark JSON 文件（顶层为记录数组）")
    parser.add_argument("--input-jsonl", default="", help="本地 benchmark.jsonl")
    parser.add_argument("--input-csv", default="", help="本地输入 CSV")
    parser.add_argument("--limit", type=int, default=0, help="最多评测 N 条")
    parser.add_argument("--output-dir", default="", help="输出目录，默认 output/multiturn_judge/run_xxx")
    parser.add_argument("--output-jsonl", default="judge_results.jsonl")
    parser.add_argument("--output-csv", default="judge_results.csv")
    parser.add_argument("--output-report-json", default="judge_report.json")
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE, help="输出 judge DW 表，留空不写")
    parser.add_argument("--output-dw-json", default="", help="把原本写入 DW 的记录（含 ds 字段）另存为该 JSON 文件路径；运行时可选指定")
    parser.add_argument("--output-partition", default="", help="输出分区，默认从 input partition 派生或当天")
    parser.add_argument("--output-overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-write-dw", action="store_true", help="跳过写 DW")
    parser.add_argument("--dump-prompts-only", action="store_true", help="只输出 judge prompt/input，不调用 LLM")

    parser.add_argument("--api-url", default=os.getenv("MULTITURN_QUALITY_JUDGE_API_URL", os.getenv("LLM_API_URL", "https://your-llm-gateway.example.com/api/v1/chat/completions")))
    parser.add_argument("--api-key", default=os.getenv("MULTITURN_QUALITY_JUDGE_API_KEY", os.getenv("LLM_API_KEY", "")))
    parser.add_argument("--model", default=os.getenv("MULTITURN_QUALITY_JUDGE_MODEL", os.getenv("LLM_MODEL_NAME", "gpt-4o")))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--request-delay-seconds", type=float, default=1, help="每条 judge 后的主动等待秒数，低并发时用于减轻限流")
    parser.add_argument("--include-system-messages", action="store_true", help="judge 输入中保留原模型 system prompt，默认不保留以节省 token")
    parser.add_argument("--max-message-chars", type=int, default=6000)
    parser.add_argument("--max-tool-chars", type=int, default=2000)
    parser.add_argument(
        "--judge-card-mode",
        choices=["auto", "daily", "offline"],
        default="auto",
        help="judge 输入的商品卡还原方式：auto/daily 会参考 online card render helper 的 Card 补商品逻辑；offline 只用 benchmark/Raw 已有信息",
    )
    parser.add_argument("--daily-preview-root", default=os.getenv("ONLINE_CARD_RENDER_ROOT", "/path/to/online_card_render"), help="online card render helper.py 所在目录")
    parser.add_argument("--preview-cookie", default=os.getenv("PREVIEW_COOKIE", ""), help="daily preview 卡片补商品时使用的 Cookie")
    parser.add_argument("--preview-cookie-file", default="", help="从文件读取 daily preview Cookie")
    parser.add_argument("--max-card-items-for-judge", type=int, default=3, help="每张商品卡传给 judge 的商品数")
    parser.add_argument("--max-product-images-per-card", type=int, default=3, help="每张商品卡最多附加多少张商品图给多模态 judge")
    parser.add_argument("--offline-manifest-path", default=os.getenv("OFFLINE_ITEM_MANIFEST", _OFFLINE_MANIFEST_PATH), help="拿到 item_id 后补全商品信息（title/price/shop_name/主图/item_pv_edit/item_ocr_info/item_brand_name）用的本地 manifest jsonl")
    parser.add_argument("--fetch-item-pv", action=argparse.BooleanOptionalAction, default=True, help="仅对【目标轮】商品，通过 item_attr_completion API 补全 item_ocr_info/item_pv_edit 并喂给 judge；历史轮不调用")
    parser.add_argument("--item-pv-fields", default=",".join(_ITEM_COMPLETION_FIELDS), help="目标轮商品补全字段，逗号分隔（默认 item_ocr_info,item_pv_edit）")
    parser.add_argument("--item-pv-url", default=_ITEM_PV_URL, help="item_attr_completion API 地址")
    parser.add_argument("--item-pv-appid", default="00000", help="item_attr_completion appid")
    parser.add_argument("--item-pv-code", default="item_attr_completion", help="item_attr_completion code")
    parser.add_argument("--item-pv-tool-call-id", default="judge_item_pv", help="item_attr_completion toolCallId")
    parser.add_argument("--item-pv-timeout", type=int, default=10, help="item_pv_edit 请求超时秒数")
    parser.add_argument("--item-pv-max-chars", type=int, default=600, help="每个商品 item_pv_edit 传给 judge 的最大字符数，0 表示不截断")
    parser.add_argument("--save-prompt-messages", action="store_true", help="在本地结果里保存 judge prompt/input，便于排查")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="从 checkpoint 续跑，跳过已有结果的 sample_id")

    parser.add_argument("--dw-access-id", default=os.getenv("DW_ACCESS_ID", ""))
    parser.add_argument("--dw-access-key", default=os.getenv("DW_ACCESS_KEY", ""))
    parser.add_argument("--dw-project", default=os.getenv("DW_PROJECT", "your_dw_project"))
    parser.add_argument("--dw-endpoint", default=os.getenv("DW_ENDPOINT", "http://dw-endpoint.example.com/api"))
    return parser.parse_args()


def prompt_dump_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    dumped = []
    for row in rows:
        payload = build_judge_payload(row, args)
        dumped.append(
            {
                "sample_id": row.get("sample_id", ""),
                "judge_prompt_messages_json": safe_json_dumps(build_judge_messages(row, args, payload=payload)),
                "judge_input_payload_json": safe_json_dumps(payload),
                "judge_success": "未执行",
            }
        )
    return dumped


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("没有可评测的输入样本")

    output_dir = Path(args.output_dir) if args.output_dir else Path("output") / "multiturn_judge" / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dump_prompts_only:
        judged = prompt_dump_rows(rows, args)
    else:
        if not args.api_key:
            raise ValueError("运行 LLM judge 需要 --api-key 或 MULTITURN_QUALITY_JUDGE_API_KEY/LLM_API_KEY")
        judged = run_judge(rows, args, output_dir=output_dir)

    judge_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in judged:
        row["judge_time"] = judge_time

    jsonl_path = output_dir / args.output_jsonl
    csv_path = output_dir / args.output_csv
    report_path = output_dir / args.output_report_json
    write_jsonl(str(jsonl_path), judged)
    write_csv(str(csv_path), judged)

    report = build_report(judged)
    report.update({"output_dir": str(output_dir), "jsonl_path": str(jsonl_path), "csv_path": str(csv_path)})

    if not args.dump_prompts_only:
        if not args.output_partition:
            input_part = partition_value(normalize_partition_spec(args.input_partition)) if args.input_partition else dt.datetime.now().strftime("%Y%m%d")
            args.output_partition = f"{input_part}_find_item_judge"
        partition_ds = partition_value(normalize_partition_spec(args.output_partition))


        if args.output_dw_json:
            write_dw_records_json(judged, args.output_dw_json, partition_ds)
            report["dw_json_path"] = args.output_dw_json
            report["dw_json_partition_ds"] = partition_ds
            args.no_write_dw = True

        write_to_dw(judged, args)
        report["dw_output_table"] = "" if args.no_write_dw else args.output_table
        report["dw_output_partition"] = "" if args.no_write_dw else args.output_partition

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
