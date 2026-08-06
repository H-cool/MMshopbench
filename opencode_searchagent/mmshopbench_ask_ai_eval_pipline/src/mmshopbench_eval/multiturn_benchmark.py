

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from mmshopbench_eval.rendering.stream_normalizer import normalize_product, normalize_url, strip_html_tags


HUMAN_REVIEW_COLUMNS = [
    "sample_id",
    "session_id",
    "target_turn_id",
    "turn_index",
    "request_time",
    "chat_round",
    "multi_turn_type",
    "context_dependency",
    "is_key_turn",
    "selector_source",
    "key_turn_reason",
    "expected_context",
    "history_user_assistant",
    "current_user_input",
    "model_output",
    "output_text",
    "output_cards",
    "output_followups",
    "上下文继承是否正确",
    "未继承的上下文点",
    "是否把当前轮误判为新任务",
    "文本问题类型",
    "文本打分",
    "文本问题备注",
    "是否出现商卡",
    "商卡问题类型",
    "外展商卡打分",
    "站内是否有供给（出卡不合理/文案证伪时）",
    "商品id",
    "外展商卡问题备注",
    "商卡跳转前4坑相关性（与query对比）",
    "引导词问题类型",
    "引导词打分",
    "引导词问题备注",
    "整体拟合打总分",
    "整体打分备注",
    "标注人",
]


BENCHMARK_JSON_COLUMNS = [
    "sample_id",
    "session_id",
    "target_turn_id",
    "turn_index",
    "request_time",
    "chat_round",
    "multi_turn_type",
    "context_dependency",
    "is_key_turn",
    "selector_source",
    "key_turn_reason",
    "expected_context",
    "history_user_assistant",
    "current_user_input",
    "model_output",
    "output_text",
    "output_cards_json",
    "output_followups_json",
    "source_session_json",
    "source_target_turn_json",
    "created_at",
]


CONTEXT_KEYWORDS = {
    "指代追问": [
        "这个",
        "这款",
        "这种",
        "刚才",
        "上面",
        "前面",
        "它",
        "那个",
        "这件",
        "这双",
        "这条",
        "这个商品",
    ],
    "追加约束": [
        "我要",
        "想要",
        "有没有",
        "带",
        "不要",
        "尺寸",
        "尺码",
        "品牌",
        "材质",
        "适合",
        "女款",
        "男款",
        "儿童",
    ],
    "纠错": [
        "不是",
        "不对",
        "错了",
        "我说的是",
        "看错",
        "别的",
        "不是这个",
    ],
    "换色/换价/换款": [
        "换",
        "黑色",
        "白色",
        "红色",
        "蓝色",
        "绿色",
        "便宜",
        "贵",
        "低价",
        "高端",
        "同款",
        "相似款",
        "小一点",
        "大一点",
    ],
    "继续找": [
        "还有吗",
        "还有没有",
        "再找",
        "继续",
        "类似",
        "相似",
        "更多",
        "再看看",
        "换一批",
    ],
    "任务切换": [
        "另外",
        "另一个",
        "新的",
        "再问",
        "这个图",
        "这张图",
    ],
}

STRONG_CONTEXT_TYPES = {"指代追问", "追加约束", "纠错", "换色/换价/换款", "继续找"}

PRIVACY_PATTERNS = [
    re.compile(r"1[3-9]\d{9}"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{17}[\dXx]\b"),
]

_CN_CHAR_RE = re.compile(r"[一-鿿]")

CARD_RE = re.compile(r"<Card>(.*?)</Card>", re.DOTALL)

MODEL_OUTPUT_EMPTY_SENTINELS = {
    "[ContentItem({'text': ''})]",
    '[ContentItem({"text": ""})]',
    "[]",
}


@dataclass
class MultiturnTurn:
    

    session_id: str
    turn_id: str
    turn_index: int
    request_time: str = ""
    chat_round: str = ""
    user_text: str = ""
    image_urls: list[str] = field(default_factory=list)
    assistant_text: str = ""
    assistant_blocks: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiturnSample:
    

    sample_id: str
    session_id: str
    target_turn_id: str
    turn_index: int
    request_time: str
    chat_round: str
    multi_turn_type: str
    context_dependency: str
    expected_context: str
    history_user_assistant: str
    current_user_input: str
    model_output: str
    output_text: str
    output_cards: list[dict[str, Any]]
    output_followups: list[str]
    source_session: dict[str, Any]
    source_target_turn: dict[str, Any]
    created_at: str = ""
    is_key_turn: bool = True
    selector_source: str = "rule"
    key_turn_reason: str = ""

    def to_benchmark_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "session_id": self.session_id,
            "target_turn_id": self.target_turn_id,
            "turn_index": self.turn_index,
            "request_time": self.request_time,
            "chat_round": self.chat_round,
            "multi_turn_type": self.multi_turn_type,
            "context_dependency": self.context_dependency,
            "is_key_turn": self.is_key_turn,
            "selector_source": self.selector_source,
            "key_turn_reason": self.key_turn_reason,
            "expected_context": self.expected_context,
            "history_user_assistant": self.history_user_assistant,
            "current_user_input": self.current_user_input,
            "model_output": self.model_output,
            "output_text": self.output_text,
            "output_cards_json": safe_json_dumps(self.output_cards),
            "output_followups_json": safe_json_dumps(self.output_followups),
            "source_session_json": safe_json_dumps(self.source_session),
            "source_target_turn_json": safe_json_dumps(self.source_target_turn),
            "created_at": self.created_at,
        }

    def to_human_review_record(self) -> dict[str, Any]:
        record = {
            "sample_id": self.sample_id,
            "session_id": self.session_id,
            "target_turn_id": self.target_turn_id,
            "turn_index": self.turn_index,
            "request_time": self.request_time,
            "chat_round": self.chat_round,
            "multi_turn_type": self.multi_turn_type,
            "context_dependency": self.context_dependency,
            "is_key_turn": "是" if self.is_key_turn else "否",
            "selector_source": self.selector_source,
            "key_turn_reason": self.key_turn_reason,
            "expected_context": self.expected_context,
            "history_user_assistant": self.history_user_assistant,
            "current_user_input": self.current_user_input,
            "model_output": self.model_output,
            "output_text": self.output_text,
            "output_cards": format_cards(self.output_cards),
            "output_followups": " / ".join(self.output_followups),
        }
        for column in HUMAN_REVIEW_COLUMNS:
            record.setdefault(column, "")
        return record


def safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def safe_json_dumps(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def truncate_text(text: str, limit: int = 240) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = first_nonempty(item.get("text"), item.get("content"), item.get("value"))
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        return first_nonempty(content.get("text"), content.get("content"), content.get("value"))
    return ""


def normalize_image_urls(value: Any) -> list[str]:
    parsed = safe_json_loads(value)
    if isinstance(parsed, list):
        urls: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                urls.append(item.strip())
            elif isinstance(item, dict):
                urls.append(first_nonempty(item.get("url"), item.get("img"), item.get("image_url"), item.get("imageUrl")))
        return [u for u in urls if u]
    if isinstance(parsed, dict):
        urls = []
        for key in ("url", "img", "image_url", "imageUrl"):
            if parsed.get(key):
                urls.append(str(parsed[key]).strip())
        if parsed.get("pics"):
            urls.extend(normalize_image_urls(parsed.get("pics")))
        if parsed.get("images"):
            urls.extend(normalize_image_urls(parsed.get("images")))
        return [u for u in urls if u]
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;\s]+", text) if part.strip() and ("://" in part or part.startswith("//"))]


def normalize_blocks(value: Any) -> list[dict[str, Any]]:
    parsed = safe_json_loads(value)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("blocks", "assistant_blocks", "surface_blocks"):
            blocks = normalize_blocks(parsed.get(key))
            if blocks:
                return blocks
    return []


def parse_model_output(raw: Any) -> str:
    

    if raw is None:
        return ""
    if isinstance(raw, list):
        parts = raw
    elif isinstance(raw, str):
        parsed = safe_json_loads(raw)
        parts = parsed if isinstance(parsed, list) else [raw]
    else:
        parts = [raw]

    kept: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = extract_text_from_content(part)
            if not text:
                text = safe_json_dumps(part)
        else:
            text = str(part or "")
        if not text.strip():
            continue
        if text.strip() in MODEL_OUTPUT_EMPTY_SENTINELS:
            continue
        kept.append(text)
    return "".join(kept)


def split_model_output_segments(full_text: str) -> list[dict[str, Any]]:
    

    segments: list[dict[str, Any]] = []
    last_end = 0
    for match in CARD_RE.finditer(str(full_text or "")):
        before = full_text[last_end : match.start()]
        if before.strip():
            segments.append({"kind": "text", "content": before})

        raw_json = match.group(1).strip()
        try:
            card = json.loads(raw_json)
            call_name = str(card.get("call_name") or "").strip()
            params = card.get("input_parameters") or {}
            if not isinstance(params, dict):
                params = {}
            segments.append(
                {
                    "kind": "card",
                    "call_name": call_name,
                    "params": params,
                    "raw": card,
                }
            )
        except Exception:
            segments.append({"kind": "text", "content": match.group(0)})
        last_end = match.end()

    trailing = full_text[last_end:]
    if trailing.strip():
        segments.append({"kind": "text", "content": trailing})
    return segments


def strip_card_protocol(text: str) -> str:
    parts = [
        str(segment.get("content") or "").strip()
        for segment in split_model_output_segments(text)
        if segment.get("kind") == "text"
    ]
    return "\n\n".join(part for part in parts if part).strip()


def normalize_product_for_preview(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        normalized = normalize_product(item)
        item_id = first_nonempty(normalized.get("id"), item.get("id"), item.get("itemId"), item.get("item_id"), item.get("nid"))
        title = first_nonempty(
            "" if normalized.get("title") == "未知商品" else normalized.get("title"),
            item.get("title"),
            item.get("name"),
            item.get("itemTitle"),
            item.get("item_title"),
            item.get("auctionTitle"),
        )
        price = first_nonempty(normalized.get("price"), item.get("price"), item.get("priceWap"))
        image = first_nonempty(
            normalized.get("image"),
            item.get("image"),
            item.get("image_url"),
            item.get("picUrl"),
            item.get("pic_url"),
            item.get("pic_path"),
        )
        detail_url = first_nonempty(normalized.get("detail_url"), item.get("detail_url"), item.get("auctionURL"))
        shop_name = first_nonempty(normalized.get("shop_name"), item.get("shop_name"), item.get("shopName"))
        sales_text = first_nonempty(normalized.get("sales_text"), item.get("sales_text"), item.get("realSales"), item.get("purchaseInfo"))
        shopping_tips = first_present(normalized.get("shoppingTips"), item.get("shoppingTips"))
    else:
        item_id = str(item or "").strip()
        title = ""
        price = ""
        image = ""
        detail_url = ""
        shop_name = ""
        sales_text = ""
        shopping_tips = ""

    if not item_id and not title:
        return None
    if not title and item_id:
        title = f"商品 {item_id}"
    if image:
        image = normalize_url(str(image))
    if detail_url:
        detail_url = normalize_url(str(detail_url))
    elif item_id:
        detail_url = f"https://product.example.com/id={item_id}"
    return {
        "id": item_id,
        "title": strip_html_tags(str(title or "未知商品")),
        "price": str(price or ""),
        "image": image,
        "shop_name": str(shop_name or ""),
        "sales_text": str(sales_text or ""),
        "detail_url": detail_url,
        "shoppingTips": shopping_tips or "",
    }


def normalize_show_items(value: Any) -> list[dict[str, Any]]:
    parsed = safe_json_loads(value)
    if isinstance(parsed, list):
        items: list[dict[str, Any]] = []
        for item in parsed:
            normalized = normalize_product_for_preview(item)
            if normalized:
                items.append(normalized)
        return items
    if isinstance(parsed, dict):
        for key in ("items", "item_list", "show_item_list", "data", "result"):
            items = normalize_show_items(parsed.get(key))
            if items:
                return items
    return []


def dedupe_products(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        normalized = normalize_product_for_preview(item)
        if not normalized:
            continue
        key = (str(normalized.get("id") or ""), str(normalized.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _merge_product(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key == "title" and str(merged.get("title") or "").startswith("商品 "):
            merged[key] = value
            continue
        if not merged.get(key) and value:
            merged[key] = value
    return merged


def product_has_display_details(item: dict[str, Any]) -> bool:
    title = str(item.get("title") or "").strip()
    if title and not title.startswith("商品 "):
        return True
    return bool(item.get("price") or item.get("image") or item.get("shop_name") or item.get("sales_text"))


def _dedupe_normalized(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("id") or ""), str(item.get("title") or ""))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def enrich_products(primary: list[dict[str, Any]], fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = dedupe_products(primary)
    fallback = dedupe_products(fallback)
    fallback_by_id = {str(item.get("id") or ""): item for item in fallback if item.get("id")}
    if primary:
        enriched = []
        for item in primary:
            item_id = str(item.get("id") or "")
            enriched.append(_merge_product(item, fallback_by_id.get(item_id, {})) if item_id else item)
        enriched = _dedupe_normalized(enriched)
        if fallback and not any(product_has_display_details(item) for item in enriched):
            return _dedupe_normalized(fallback + enriched)
        return enriched
    return fallback


def parse_products_from_tool_text(text: str) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    key_map = {
        "item_id": "item_id",
        "itemid": "item_id",
        "id": "item_id",
        "title": "title",
        "price": "price",
        "image": "image",
        "image_url": "image",
        "pic": "image",
        "shop_name": "shop_name",
        "shop": "shop_name",
    }

    def flush() -> None:
        nonlocal current
        normalized = normalize_product_for_preview(current)
        if normalized:
            products.append(normalized)
        current = {}

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:：]\s*(.*)$", line)
        if not match:
            continue
        raw_key, value = match.group(1).lower(), match.group(2).strip()
        key = key_map.get(raw_key)
        if not key:
            continue
        if key == "item_id" and current.get("item_id"):
            flush()
        current[key] = value
    flush()
    return dedupe_products(products)


def extract_products_from_raw_log(raw_log: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(raw_log, dict):
        return []
    prompt = raw_log.get("prompt")
    messages = safe_json_loads(prompt)
    products: list[dict[str, Any]] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            products.extend(parse_products_from_tool_text(str(message.get("content") or "")))
    return dedupe_products(products)


def filter_products_by_ids(items: list[dict[str, Any]], item_ids: list[Any]) -> list[dict[str, Any]]:
    wanted = [str(item_id).strip() for item_id in item_ids if str(item_id or "").strip()]
    if not wanted:
        return items
    by_id = {str(item.get("id") or ""): item for item in items if item.get("id")}
    filtered: list[dict[str, Any]] = []
    for item_id in wanted:
        filtered.append(by_id.get(item_id) or normalize_product_for_preview(item_id) or {})
    return [item for item in filtered if item]


def _card_display_title(params: dict[str, Any]) -> str:
    title = first_nonempty(params.get("title"), params.get("cateTitle"), params.get("display_title"))
    if title:
        return title
    queries = params.get("search") or params.get("queries") or []
    if isinstance(queries, str):
        return queries
    if isinstance(queries, list):
        return first_nonempty(*queries)
    return ""


def build_blocks_from_model_output(
    model_output: Any,
    *,
    show_items: Any = None,
    raw_log: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    

    full_text = parse_model_output(model_output)
    show_products = normalize_show_items(show_items)
    tool_products = extract_products_from_raw_log(raw_log)
    card_products = enrich_products(show_products, tool_products)
    blocks: list[dict[str, Any]] = []
    seen_card_sigs: set[tuple[Any, ...]] = set()

    for seq, segment in enumerate(split_model_output_segments(full_text)):
        if segment.get("kind") == "text":
            content = str(segment.get("content") or "").strip()
            if content:
                blocks.append({"seq": seq, "type": "markdown", "content": content})
            continue

        call_name = str(segment.get("call_name") or "")
        params = segment.get("params") if isinstance(segment.get("params"), dict) else {}
        if call_name in {"FollowUpQuestionsWithInputCard", "FollowUpQuestionsCard"}:
            questions = params.get("questions") or params.get("queries") or []
            if isinstance(questions, str):
                questions = [questions]
            questions = [str(question).strip() for question in questions if str(question or "").strip()]
            sig = ("follow_up_questions_card", tuple(questions))
            if questions and sig not in seen_card_sigs:
                seen_card_sigs.add(sig)
                blocks.append(
                    {
                        "seq": seq,
                        "type": "follow_up_questions_card",
                        "title": first_nonempty(params.get("title"), "你还可以问"),
                        "questions": questions,
                    }
                )
            continue

        if call_name == "ItemListCard":
            items = filter_products_by_ids(card_products, params.get("item_ids") or params.get("itemIds") or [])
            sig = ("item_list_card", tuple(str(item.get("id") or item.get("title") or "") for item in items))
            if items and sig not in seen_card_sigs:
                seen_card_sigs.add(sig)
                blocks.append(
                    {
                        "seq": seq,
                        "type": "item_list_card",
                        "expand_text": first_nonempty(params.get("expand_text"), "查看更多商品"),
                        "items": items,
                    }
                )
            continue

        if call_name == "ItemFeedsCard":
            queries = params.get("search") or params.get("queries") or []
            if isinstance(queries, str):
                queries = [queries]
            display_title = _card_display_title(params)
            sig = ("item_feeds_card", tuple(str(query) for query in queries), display_title)
            if card_products and sig not in seen_card_sigs:
                seen_card_sigs.add(sig)
                blocks.append(
                    {
                        "seq": seq,
                        "type": "item_feeds_card",
                        "display_title": display_title,
                        "expand_text": first_nonempty(params.get("expand_text"), "查看更多商品"),
                        "first_show_count": 1,
                        "items": card_products,
                    }
                )
            continue

        if call_name == "ImageSameItemsCard":
            sig = ("image_same_items_card", tuple(str(item.get("id") or item.get("title") or "") for item in card_products))
            if card_products and sig not in seen_card_sigs:
                seen_card_sigs.add(sig)
                blocks.append(
                    {
                        "seq": seq,
                        "type": "image_same_items_card",
                        "expand_text": first_nonempty(params.get("expand_text"), "查看更多同款"),
                        "items": card_products,
                    }
                )
            continue

        if call_name == "MultiRegionImageSameItemsCard":
            sig = ("multi_region_image_same_items_card", tuple(str(item.get("id") or item.get("title") or "") for item in card_products))
            if card_products and sig not in seen_card_sigs:
                seen_card_sigs.add(sig)
                blocks.append(
                    {
                        "seq": seq,
                        "type": "multi_region_image_same_items_card",
                        "items": card_products,
                    }
                )

    if full_text and not blocks:
        text = strip_card_protocol(full_text) or full_text
        blocks.append({"seq": 0, "type": "markdown", "content": text})
    if card_products and not any("card" in str(block.get("type") or "") for block in blocks):
        blocks.append(
            {
                "seq": len(blocks),
                "type": "item_feeds_card",
                "display_title": "",
                "expand_text": "查看更多商品",
                "first_show_count": 1,
                "items": card_products,
            }
        )
    return blocks


def blocks_need_model_output_parse(blocks: list[dict[str, Any]]) -> bool:
    if not blocks:
        return True
    for block in blocks:
        if block.get("type") != "markdown":
            continue
        content = str(block.get("content") or "").strip()
        if "<Card>" in content or content.startswith("["):
            return True
    return False


def rebuild_blocks_for_turn(raw_turn: dict[str, Any], existing_blocks: Any = None) -> list[dict[str, Any]]:
    blocks = normalize_blocks(existing_blocks)
    if raw_turn.get("_blocks_final") and blocks and not blocks_need_model_output_parse(blocks):
        return blocks
    blocks = normalize_blocks(existing_blocks)
    source_row = raw_turn.get("source_row") if isinstance(raw_turn.get("source_row"), dict) else {}
    raw_log = first_present(raw_turn.get("raw_log"), source_row.get("raw_log"))
    model_output = first_present(
        raw_turn.get("model_output"),
        raw_turn.get("assistant_text"),
        raw_turn.get("response"),
        source_row.get("model_output"),
        raw_log.get("response") if isinstance(raw_log, dict) else "",
    )
    show_items = first_present(raw_turn.get("show_item_list"), source_row.get("show_item_list"))
    if model_output and blocks_need_model_output_parse(blocks):
        rebuilt = build_blocks_from_model_output(
            model_output,
            show_items=show_items,
            raw_log=raw_log if isinstance(raw_log, dict) else None,
        )
        if rebuilt:
            return rebuilt
    return blocks


def extract_output_cards(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for block in blocks:
        btype = str(block.get("type") or "")
        if "card" not in btype:
            continue
        if btype == "follow_up_questions_card":
            continue
        items = block.get("items") or block.get("data") or []
        if not isinstance(items, list):
            items = []
        preview_items = []
        for item in items[:4]:
            if not isinstance(item, dict):
                continue
            preview_items.append(
                {
                    "id": first_nonempty(item.get("id"), item.get("item_id"), item.get("itemId")),
                    "title": first_nonempty(item.get("title"), item.get("name")),
                    "price": first_nonempty(item.get("price"), item.get("priceWap")),
                }
            )
        cards.append({"type": btype, "count": len(items), "items": preview_items})
    return cards


def extract_followups(blocks: list[dict[str, Any]]) -> list[str]:
    followups: list[str] = []
    for block in blocks:
        btype = str(block.get("type") or "")
        if "follow" not in btype and "question" not in btype:
            continue
        questions = block.get("questions") or block.get("items") or []
        if isinstance(questions, list):
            followups.extend(str(q.get("text") if isinstance(q, dict) else q).strip() for q in questions)
    return [q for q in followups if q]


def format_cards(cards: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for card in cards:
        lines.append(f"{card.get('type', '')} 共{card.get('count', 0)}件")
        for item in card.get("items") or []:
            title = truncate_text(str(item.get("title") or ""), 60)
            price = item.get("price") or ""
            item_id = item.get("id") or ""
            lines.append(f"- {item_id} {title} {price}".strip())
    return "\n".join(lines)


def assistant_output_text(turn: MultiturnTurn) -> str:
    text = strip_card_protocol(parse_model_output(turn.assistant_text))
    if text:
        return text
    lines = []
    for block in turn.assistant_blocks:
        if block.get("type") == "markdown":
            content = strip_card_protocol(parse_model_output(block.get("content")))
            if content:
                lines.append(content)
    return "\n".join(lines).strip()


def format_user_input(turn: MultiturnTurn) -> str:
    parts = []
    if turn.image_urls:
        parts.append("图片: " + " | ".join(turn.image_urls))
    if turn.user_text:
        parts.append("文字: " + turn.user_text)
    return "\n".join(parts) if parts else "（无用户输入）"


def format_model_output(turn: MultiturnTurn) -> str:
    blocks = turn.assistant_blocks
    cards = extract_output_cards(blocks)
    followups = extract_followups(blocks)
    parts = []
    text = assistant_output_text(turn)
    if text:
        parts.append("[文本]\n" + text)
    if cards:
        parts.append("[商卡]\n" + format_cards(cards))
    if followups:
        parts.append("[引导词]\n" + " / ".join(followups))
    return "\n\n".join(parts) if parts else "（无模型输出）"


def format_history(turns: list[MultiturnTurn]) -> str:
    chunks: list[str] = []
    for turn in turns:
        chunks.append(f"=== 第{turn.turn_index}轮 ===")
        chunks.append("[用户]\n" + format_user_input(turn))
        chunks.append("[AI]\n" + format_model_output(turn))
    return "\n\n".join(chunks)


def _keyword_in_text(keyword: str, text: str) -> bool:
    
    if len(keyword) != 1:
        return keyword in text
    for m in re.finditer(re.escape(keyword), text):
        i = m.start()
        prev_cn = i > 0 and bool(_CN_CHAR_RE.match(text[i - 1]))
        next_cn = i + 1 < len(text) and bool(_CN_CHAR_RE.match(text[i + 1]))
        if not (prev_cn and next_cn):
            return True
    return False


def classify_multi_turn_type(user_text: str, image_urls: list[str] | None = None) -> str:
    text = str(user_text or "").strip()
    matched: list[str] = []
    for label, keywords in CONTEXT_KEYWORDS.items():
        if any(_keyword_in_text(keyword, text) for keyword in keywords):
            matched.append(label)
    if "纠错" in matched:
        return "纠错"
    if "换色/换价/换款" in matched:
        return "换色/换价/换款"
    if "继续找" in matched:
        return "继续找"
    if "指代追问" in matched:
        return "指代追问"
    if "追加约束" in matched:
        return "追加约束"
    if "任务切换" in matched or (image_urls and text):
        return "任务切换"
    return "弱多轮追问"


def infer_context_dependency(multi_turn_type: str, user_text: str, image_urls: list[str] | None = None) -> str:
    text = str(user_text or "").strip()
    if multi_turn_type in STRONG_CONTEXT_TYPES:
        return "强依赖历史"
    if not image_urls and len(text) <= 20:
        return "强依赖历史"
    return "弱依赖历史"


def build_expected_context(history: list[MultiturnTurn], target: MultiturnTurn, multi_turn_type: str) -> str:
    anchor = None
    for turn in reversed(history):
        if turn.user_text or turn.image_urls:
            anchor = turn
            break
    anchor_text = ""
    if anchor:
        anchor_parts = []
        if anchor.image_urls:
            anchor_parts.append(f"第{anchor.turn_index}轮图片主体")
        if anchor.user_text:
            anchor_parts.append(f"第{anchor.turn_index}轮用户文字: {truncate_text(anchor.user_text, 120)}")
        ai_text = truncate_text(assistant_output_text(anchor), 120)
        if ai_text:
            anchor_parts.append(f"第{anchor.turn_index}轮AI结论: {ai_text}")
        anchor_text = "；".join(anchor_parts)
    if not anchor_text:
        anchor_text = "最近一轮有效用户输入"
    current = truncate_text(target.user_text, 120) or "当前轮无文字，仅根据历史和图片判断"
    return f"自动预标：{multi_turn_type}。本轮应继承 {anchor_text}；并结合当前轮约束/追问：{current}"


def has_privacy_risk(turns: Iterable[MultiturnTurn]) -> bool:
    for turn in turns:
        text = "\n".join([turn.user_text, turn.assistant_text, safe_json_dumps(turn.assistant_blocks)])
        if any(pattern.search(text) for pattern in PRIVACY_PATTERNS):
            return True
    return False


def turn_has_assistant_output(turn: MultiturnTurn) -> bool:
    return bool(assistant_output_text(turn) or extract_output_cards(turn.assistant_blocks) or extract_followups(turn.assistant_blocks))


def normalize_session_turns(session: dict[str, Any]) -> list[MultiturnTurn]:
    session_id = first_nonempty(session.get("session_id"), session.get("source_name"), session.get("id"), session.get("trace_id"))
    raw_turns = (
        session.get("turns")
        or session.get("messages")
        or session.get("history")
        or session.get("conversation")
        or []
    )
    if not isinstance(raw_turns, list):
        return []

    turns: list[MultiturnTurn] = []
    for index, raw_turn in enumerate(raw_turns, start=1):
        if not isinstance(raw_turn, dict):
            continue
        result_json = safe_json_loads(raw_turn.get("result_json")) or {}
        block_source = first_present(raw_turn.get("blocks"), raw_turn.get("assistant_blocks"), raw_turn.get("surface_blocks"))
        blocks = normalize_blocks(block_source) or normalize_blocks(result_json)
        user_content = raw_turn.get("content") if raw_turn.get("role") == "user" else None
        assistant_content = raw_turn.get("content") if raw_turn.get("role") == "assistant" else None
        user_text = first_nonempty(
            raw_turn.get("user_text"),
            raw_turn.get("input_query"),
            raw_turn.get("query"),
            raw_turn.get("text"),
            extract_text_from_content(user_content),
            result_json.get("input_query") if isinstance(result_json, dict) else "",
        )
        image_urls = []
        for value in (
            raw_turn.get("image_urls"),
            raw_turn.get("images"),
            raw_turn.get("pics"),
            raw_turn.get("image_url"),
            raw_turn.get("input_image_url"),
            result_json.get("input_image_url") if isinstance(result_json, dict) else "",
            user_content,
        ):
            image_urls.extend(normalize_image_urls(value))
        image_urls = list(dict.fromkeys(image_urls))
        assistant_text = first_nonempty(
            raw_turn.get("assistant_text"),
            raw_turn.get("final_text"),
            raw_turn.get("response"),
            raw_turn.get("model_output"),
            extract_text_from_content(assistant_content),
            result_json.get("final_text") if isinstance(result_json, dict) else "",
        )
        blocks = rebuild_blocks_for_turn(
            {
                **raw_turn,
                "assistant_text": assistant_text,
                "source_row": raw_turn.get("source_row") or raw_turn,
            },
            blocks,
        )
        turn_id = first_nonempty(raw_turn.get("turn_id"), raw_turn.get("id"), raw_turn.get("message_id"), str(index))
        request_time = str(raw_turn.get("request_time") or raw_turn.get("req_time") or "").strip()
        chat_round = str(raw_turn.get("chat_round") or raw_turn.get("round") or "").strip()
        turn_index = int(float(raw_turn.get("turn_index") or raw_turn.get("chat_round") or raw_turn.get("round") or index))
        turns.append(
            MultiturnTurn(
                session_id=session_id,
                turn_id=turn_id,
                turn_index=turn_index,
                request_time=request_time,
                chat_round=chat_round,
                user_text=user_text,
                image_urls=image_urls,
                assistant_text=assistant_text,
                assistant_blocks=blocks,
                raw=raw_turn,
            )
        )
    return sorted(turns, key=lambda turn: (turn.turn_index, turn.turn_id))


def _sort_value(value: Any) -> tuple[int, float, str]:
    text = str(value or "").strip()
    if not text:
        return (1, 0.0, "")
    try:
        return (0, float(text), text)
    except Exception:
        return (0, 0.0, text)


def pv_rows_to_sessions(
    rows: list[dict[str, Any]],
    *,
    raw_by_trace: dict[str, dict[str, Any]] | None = None,
    conversation_id_col: str = "conversation_id",
    chat_id_col: str = "chat_id",
    trace_id_col: str = "trace_id",
    image_url_col: str = "query_pic_img",
    query_col: str = "query_text",
    model_output_col: str = "model_output",
    show_item_list_col: str = "show_item_list",
    request_time_col: str = "request_time",
    chat_round_col: str = "chat_round",
    order_col: str = "request_time",
) -> list[dict[str, Any]]:
    

    raw_by_trace = raw_by_trace or {}
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        conversation_id = first_nonempty(
            row.get(conversation_id_col),
            row.get("session_id"),
            row.get("chat_id"),
            f"conversation_{index:06d}",
        )
        grouped[conversation_id].append((index, row))

    sessions: list[dict[str, Any]] = []
    for conversation_id, indexed_rows in grouped.items():
        ordered = sorted(
            indexed_rows,
            key=lambda item: (
                _sort_value(item[1].get(chat_round_col)),
                _sort_value(item[1].get(order_col)),
                _sort_value(item[1].get(chat_id_col)),
                item[0],
            ),
        )
        turns: list[dict[str, Any]] = []
        for turn_index, (input_index, row) in enumerate(ordered, start=1):
            trace_id = str(row.get(trace_id_col) or "").strip()
            raw_log = raw_by_trace.get(trace_id, {})
            model_output = first_nonempty(
                row.get(model_output_col),
                raw_log.get("response"),
            )
            show_items = normalize_show_items(row.get(show_item_list_col))
            raw_turn = dict(row)
            if raw_log:
                raw_turn["raw_log"] = raw_log
            raw_turn["_input_index"] = input_index
            blocks = build_blocks_from_model_output(
                model_output,
                show_items=show_items or row.get(show_item_list_col),
                raw_log=raw_log,
            )
            turns.append(
                {
                    "turn_id": first_nonempty(row.get(chat_id_col), trace_id, str(input_index)),
                    "turn_index": row.get(chat_round_col) or turn_index,
                    "request_time": row.get(request_time_col),
                    "chat_round": row.get(chat_round_col),
                    "user_text": row.get(query_col),
                    "image_url": row.get(image_url_col),
                    "assistant_text": model_output,
                    "blocks": blocks,
                    "_blocks_final": True,
                    "trace_id": trace_id,
                    "source_row": raw_turn,
                }
            )
        sessions.append({"session_id": conversation_id, "conversation_id": conversation_id, "turns": turns})
    return sessions


def rows_to_sessions(
    rows: list[dict[str, Any]],
    *,
    turns_json_col: str = "",
    session_id_col: str = "session_id",
    turn_id_col: str = "turn_id",
    turn_index_col: str = "turn_index",
    user_text_col: str = "user_text",
    image_url_col: str = "image_url",
    assistant_text_col: str = "assistant_text",
    blocks_json_col: str = "blocks_json",
    request_time_col: str = "request_time",
    chat_round_col: str = "chat_round",
) -> list[dict[str, Any]]:
    if turns_json_col:
        sessions = []
        for index, row in enumerate(rows, start=1):
            session_id = first_nonempty(row.get(session_id_col), row.get("source_name"), row.get("id"), f"session_{index:06d}")
            turns = safe_json_loads(row.get(turns_json_col)) or []
            sessions.append({"session_id": session_id, "turns": turns, "source_row": row})
        return sessions

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        session_id = first_nonempty(row.get(session_id_col), row.get("source_name"), row.get("id"), f"session_{index:06d}")
        turn = {
            "turn_id": first_nonempty(row.get(turn_id_col), row.get("id"), str(index)),
            "turn_index": row.get(turn_index_col) or index,
            "request_time": row.get(request_time_col),
            "chat_round": row.get(chat_round_col),
            "user_text": row.get(user_text_col),
            "image_url": row.get(image_url_col),
            "assistant_text": row.get(assistant_text_col),
            "blocks": row.get(blocks_json_col),
            "source_row": row,
        }
        groups[session_id].append(turn)
    return [{"session_id": sid, "turns": turns} for sid, turns in groups.items()]


def build_candidate_samples(
    sessions: list[dict[str, Any]],
    *,
    created_at: str = "",
    skip_privacy: bool = True,
) -> tuple[list[MultiturnSample], dict[str, Any]]:
    candidates: list[MultiturnSample] = []
    stats = Counter()
    created = created_at or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for session in sessions:
        stats["sessions_total"] += 1
        turns = normalize_session_turns(session)
        if len(turns) < 2:
            stats["skip_short_session"] += 1
            continue
        if skip_privacy and has_privacy_risk(turns):
            stats["skip_privacy"] += 1
            continue
        for target_pos, target in enumerate(turns[1:], start=1):
            stats["target_turns_seen"] += 1
            history = turns[:target_pos]
            if not history:
                stats["skip_no_history"] += 1
                continue
            if not turn_has_assistant_output(target):
                stats["skip_no_assistant_output"] += 1
                continue
            multi_turn_type = classify_multi_turn_type(target.user_text, target.image_urls)
            context_dependency = infer_context_dependency(multi_turn_type, target.user_text, target.image_urls)
            sample_id = f"{target.session_id}_{target.turn_index}_{target.turn_id}"
            sample = MultiturnSample(
                sample_id=sample_id,
                session_id=target.session_id,
                target_turn_id=target.turn_id,
                turn_index=target.turn_index,
                request_time=target.request_time,
                chat_round=target.chat_round,
                multi_turn_type=multi_turn_type,
                context_dependency=context_dependency,
                expected_context=build_expected_context(history, target, multi_turn_type),
                history_user_assistant=format_history(history),
                current_user_input=format_user_input(target),
                model_output=format_model_output(target),
                output_text=assistant_output_text(target),
                output_cards=extract_output_cards(target.assistant_blocks),
                output_followups=extract_followups(target.assistant_blocks),
                source_session=session,
                source_target_turn=target.raw,
                created_at=created,
            )
            candidates.append(sample)
    seen_ids: set[str] = set()
    unique_candidates: list[MultiturnSample] = []
    for sample in candidates:
        if sample.sample_id not in seen_ids:
            seen_ids.add(sample.sample_id)
            unique_candidates.append(sample)
    stats["candidates"] = len(unique_candidates)
    return unique_candidates, dict(stats)


def select_samples(
    candidates: list[MultiturnSample],
    *,
    sample_size: int = 200,
    seed: int = 42,
    min_strong_ratio: float = 0.8,
) -> list[MultiturnSample]:
    rng = random.Random(seed)
    strong = [sample for sample in candidates if sample.context_dependency == "强依赖历史"]
    weak = [sample for sample in candidates if sample.context_dependency != "强依赖历史"]
    rng.shuffle(strong)
    rng.shuffle(weak)
    if sample_size <= 0:
        sample_size = len(candidates)
    strong_target = min(len(strong), math.ceil(sample_size * min_strong_ratio))
    selected = strong[:strong_target]
    remaining = sample_size - len(selected)
    selected.extend(weak[:remaining])
    if len(selected) < sample_size:
        selected.extend(strong[strong_target : strong_target + sample_size - len(selected)])
    return selected[:sample_size]


def benchmark_report(samples: list[MultiturnSample], build_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    type_counts = Counter(sample.multi_turn_type for sample in samples)
    dependency_counts = Counter(sample.context_dependency for sample in samples)
    selector_counts = Counter(sample.selector_source for sample in samples)
    strong_count = dependency_counts.get("强依赖历史", 0)
    selected_count = len(samples)
    required_types = ["指代追问", "追加约束", "纠错", "继续找", "任务切换"]
    missing_types = [label for label in required_types if type_counts.get(label, 0) == 0]
    report = {
        "selected_count": selected_count,
        "strong_context_count": strong_count,
        "strong_context_ratio": round(strong_count / selected_count, 4) if selected_count else 0.0,
        "multi_turn_type_counts": dict(type_counts),
        "context_dependency_counts": dict(dependency_counts),
        "selector_source_counts": dict(selector_counts),
        "missing_required_types": missing_types,
    }
    if build_stats:
        report["build_stats"] = build_stats
    return report


def samples_to_jsonl(samples: list[MultiturnSample]) -> str:
    return "\n".join(safe_json_dumps(sample.to_benchmark_record()) for sample in samples) + ("\n" if samples else "")


def samples_to_human_csv(samples: list[MultiturnSample]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=HUMAN_REVIEW_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for sample in samples:
        writer.writerow(sample.to_human_review_record())
    return output.getvalue()


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "是", "关键", "关键轮"}:
        return True
    if text in {"false", "0", "no", "n", "否", "非关键", "非关键轮"}:
        return False
    return None


def apply_key_turn_decision(
    sample: MultiturnSample,
    decision: dict[str, Any],
    *,
    fallback_to_rule: bool = True,
) -> MultiturnSample:
    

    is_key_turn = parse_bool(decision.get("is_key_turn"))
    has_error = bool(decision.get("error") or decision.get("judge_error"))
    if is_key_turn is None:
        is_key_turn = sample.context_dependency == "强依赖历史" if fallback_to_rule else False

    multi_turn_type = str(decision.get("multi_turn_type") or "").strip()
    context_dependency = str(decision.get("context_dependency") or "").strip()
    expected_context = str(decision.get("expected_context") or "").strip()
    reason = first_nonempty(decision.get("reason"), decision.get("key_turn_reason"), decision.get("error"), decision.get("judge_error"))

    if multi_turn_type and multi_turn_type != "非关键轮":
        sample.multi_turn_type = multi_turn_type
    if context_dependency:
        sample.context_dependency = context_dependency
    if expected_context:
        sample.expected_context = expected_context
    sample.is_key_turn = bool(is_key_turn)
    sample.key_turn_reason = reason
    if has_error:
        sample.selector_source = "llm_error_rule_fallback" if fallback_to_rule else "llm_error"
    else:
        sample.selector_source = "llm"
    return sample


def read_csv_records(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_yes_no(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"是", "正确", "对", "1", "true", "True", "yes", "Y"}:
        return "是"
    if text in {"否", "错误", "错", "0", "false", "False", "no", "N"}:
        return "否"
    if text in {"不适用", "NA", "N/A", "-"}:
        return "不适用"
    return text


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def validate_human_review_rows(
    rows: list[dict[str, Any]],
    *,
    min_strong_ratio: float = 0.8,
    required_types: list[str] | None = None,
) -> dict[str, Any]:
    required_types = required_types or ["指代追问", "追加约束", "纠错", "继续找", "任务切换"]
    errors: list[str] = []
    warnings: list[str] = []
    type_counts = Counter(str(row.get("multi_turn_type") or "").strip() for row in rows)
    strong_count = sum(1 for row in rows if str(row.get("context_dependency") or "").strip() == "强依赖历史")
    total = len(rows)
    strong_ratio = strong_count / total if total else 0.0

    if total == 0:
        errors.append("没有可校验的样本行")
    if strong_ratio < min_strong_ratio:
        errors.append(f"强依赖历史占比 {strong_ratio:.2%} 低于阈值 {min_strong_ratio:.0%}")

    missing_types = [label for label in required_types if type_counts.get(label, 0) == 0]
    if missing_types:
        warnings.append("缺少场景类型: " + "、".join(missing_types))

    missing_overall = []
    missing_overall_note = []
    missing_context_points = []
    for index, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id") or f"row_{index}").strip()
        if parse_float(row.get("整体拟合打总分")) is None:
            missing_overall.append(sample_id)
        if not str(row.get("整体打分备注") or "").strip():
            missing_overall_note.append(sample_id)
        context_ok = normalize_yes_no(row.get("上下文继承是否正确"))
        if context_ok == "否" and not str(row.get("未继承的上下文点") or "").strip():
            missing_context_points.append(sample_id)

    if missing_overall:
        errors.append("缺少整体拟合打总分: " + ", ".join(missing_overall[:20]))
    if missing_overall_note:
        errors.append("缺少整体打分备注: " + ", ".join(missing_overall_note[:20]))
    if missing_context_points:
        errors.append("上下文继承错误但未填写未继承点: " + ", ".join(missing_context_points[:20]))

    return {
        "row_count": total,
        "strong_context_count": strong_count,
        "strong_context_ratio": round(strong_ratio, 4) if total else 0.0,
        "multi_turn_type_counts": dict(type_counts),
        "missing_required_types": missing_types,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def pass_fail(score: Any, threshold: float) -> str:
    parsed = parse_float(score)
    if parsed is None:
        return ""
    return "pass" if parsed >= threshold else "fail"


def first_badcase_type(row: dict[str, Any]) -> str:
    for key in ("文本问题类型", "商卡问题类型", "引导词问题类型"):
        value = str(row.get(key) or "").strip()
        if value and value not in {"无", "无问题", "符合预期", "正常"}:
            return value
    return ""


def compare_human_and_judge(
    human_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    *,
    pass_threshold: float = 3.0,
) -> dict[str, Any]:
    judge_by_id = {str(row.get("sample_id") or "").strip(): row for row in judge_rows}
    comparable = []
    for human in human_rows:
        sample_id = str(human.get("sample_id") or "").strip()
        judge = judge_by_id.get(sample_id)
        if judge:
            comparable.append((human, judge))

    categorical_fields = [
        "上下文继承是否正确",
        "是否把当前轮误判为新任务",
        "文本问题类型",
        "商卡问题类型",
        "引导词问题类型",
    ]
    field_accuracy: dict[str, float] = {}
    for field_name in categorical_fields:
        total = 0
        correct = 0
        for human, judge in comparable:
            h_value = normalize_yes_no(human.get(field_name)) if "是否" in field_name else str(human.get(field_name) or "").strip()
            j_value = normalize_yes_no(judge.get(field_name)) if "是否" in field_name else str(judge.get(field_name) or "").strip()
            if not h_value:
                continue
            total += 1
            correct += int(h_value == j_value)
        field_accuracy[field_name] = round(correct / total, 4) if total else 0.0

    human_scores: list[float] = []
    judge_scores: list[float] = []
    pass_total = 0
    pass_correct = 0
    badcase_total = 0
    badcase_correct = 0
    for human, judge in comparable:
        h_score = parse_float(human.get("整体拟合打总分"))
        j_score = parse_float(judge.get("整体拟合打总分"))
        if h_score is not None and j_score is not None:
            human_scores.append(h_score)
            judge_scores.append(j_score)
            pass_total += 1
            pass_correct += int(pass_fail(h_score, pass_threshold) == pass_fail(j_score, pass_threshold))
        h_badcase = first_badcase_type(human)
        if h_badcase:
            badcase_total += 1
            badcase_correct += int(h_badcase == first_badcase_type(judge))

    corr = pearson(human_scores, judge_scores)
    return {
        "human_rows": len(human_rows),
        "judge_rows": len(judge_rows),
        "matched_rows": len(comparable),
        "field_accuracy": field_accuracy,
        "overall_score_pearson": round(corr, 4) if corr is not None else None,
        "pass_fail_accuracy": round(pass_correct / pass_total, 4) if pass_total else 0.0,
        "badcase_type_accuracy": round(badcase_correct / badcase_total, 4) if badcase_total else 0.0,
        "pass_threshold": pass_threshold,
    }
