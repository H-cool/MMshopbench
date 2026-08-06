from __future__ import annotations

import json
import re
import urllib.parse


SHOP_NOISE_TEXT = {"进店", "购物平台超市", "买过的店"}
MARKDOWN_EMPTY_SENTINELS = {
    "[ContentItem({'text': ''})]",
    "[ContentItem({\"text\": \"\"})]",
    "[]",
}


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    if url.startswith("//"):
        url = "https:" + url

    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    return url


def strip_html_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def pick_image_url(item: dict) -> str:
    image_info = item.get("imageInfo") or []
    if isinstance(image_info, list):
        for info in image_info:
            if isinstance(info, dict) and info.get("imageUrl"):
                return normalize_url(info["imageUrl"])

    for key_path in (
        ("itemPic", "src"),
        ("itemPic", "url"),
        ("image",),
        ("pic_path",),
    ):
        current = item
        found = True
        for key in key_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
                break
        if found and isinstance(current, str) and current:
            return normalize_url(current)

    return ""


def pick_shop_name(item: dict) -> str:
    structured = item.get("structuredShopInfo") or {}
    info_list = structured.get("infoList") or []
    candidates = []
    if isinstance(info_list, list):
        for info in info_list:
            if not isinstance(info, dict):
                continue
            text = str(info.get("text") or "").strip()
            if not text:
                continue
            if text in SHOP_NOISE_TEXT:
                continue
            if text.endswith("老店") or text.startswith("回头客") or "搜索" in text:
                continue
            candidates.append(text)
    if candidates:
        return candidates[-1]

    shop_info = item.get("shopInfo") or {}
    shop_info_list = shop_info.get("shopInfoList") or []
    if isinstance(shop_info_list, list):
        for text in shop_info_list:
            text = str(text or "").strip()
            if text and text not in SHOP_NOISE_TEXT:
                return text

    return ""


def pick_price(item: dict) -> str:
    for key in ("priceShowWithIcon", "priceShow"):
        price_info = item.get(key) or {}
        price = str(price_info.get("price") or "").strip()
        if price:
            unit = str(price_info.get("unit") or "¥").strip()
            return f"{unit}{price}" if not price.startswith(("¥", "￥")) else price

    raw_price = str(item.get("priceWap") or item.get("price") or "").strip()
    if raw_price:
        return raw_price if raw_price.startswith(("¥", "￥")) else f"¥{raw_price}"
    return ""


def normalize_sales_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    compact = re.sub(r"\s+", "", text)


    if re.fullmatch(r"\d+(?:\.\d+)?\+?", compact):
        return ""

    return text


def pick_sales_text(item: dict) -> str:
    for key in ("realSales", "purchaseInfo"):
        value = normalize_sales_text(item.get(key))
        if value:
            return value

    price_show = item.get("priceShow") or {}
    for key in ("daySold", "sold"):
        value = normalize_sales_text(price_show.get(key))
        if value:
            return value

    return ""


def pick_item_id(item: dict) -> str:
    for key in ("itemId", "item_id", "nid"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def pick_detail_url(item: dict) -> str:
    return normalize_url(str(item.get("auctionURL") or ""))


def pick_shopping_tips(item: dict) -> str:
    values: list[str] = []

    def walk(value) -> None:
        if value in (None, ""):
            return
        if isinstance(value, str):
            text = strip_html_tags(value)
            if text:
                values.append(text)
            return
        if isinstance(value, (int, float)):
            values.append(str(value))
            return
        if isinstance(value, list):
            for sub in value:
                walk(sub)
            return
        if isinstance(value, dict):
            for key in ("text", "content", "title", "value", "name", "desc"):
                if key in value:
                    walk(value.get(key))
            for key in ("items", "tips", "list", "tags"):
                nested = value.get(key)
                if isinstance(nested, list):
                    for sub in nested:
                        walk(sub)

    walk(item.get("shoppingTips"))

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)

    return "；".join(deduped)


def normalize_product(item: dict) -> dict:
    shopping_tips = pick_shopping_tips(item)
    return {
        "id": pick_item_id(item),
        "title": strip_html_tags(str(item.get("title") or "未知商品")),
        "price": pick_price(item),
        "image": pick_image_url(item),
        "shop_name": pick_shop_name(item),
        "sales_text": pick_sales_text(item),
        "detail_url": pick_detail_url(item),
        "shoppingTips": shopping_tips,
    }


def make_items_signature(items: list[dict]) -> tuple:
    signature = []
    for item in items or []:
        if not isinstance(item, dict):
            signature.append(("__raw__", str(item)))
            continue
        signature.append((
            str(item.get("id") or ""),
            str(item.get("title") or ""),
            str(item.get("price") or ""),
            str(item.get("image") or ""),
            str(item.get("detail_url") or ""),
        ))
    return tuple(signature)


def extract_follow_up_questions(raw_items: list[dict]) -> tuple[str, list[str]]:
    first_result = raw_items[0] if raw_items else {}
    data = first_result.get("data") or {}

    title = str(data.get("title") or "你还可以问")
    questions: list[str] = []

    for item in data.get("queries") or []:
        query = str((item or {}).get("query") or "").strip()
        if query:
            questions.append(query)

    if not questions:
        for question in data.get("questions") or []:
            if isinstance(question, dict):
                text = str(
                    question.get("query")
                    or question.get("text")
                    or question.get("title")
                    or ""
                ).strip()
            else:
                text = str(question or "").strip()
            if text:
                questions.append(text)

    return title, questions


def extract_input_image(payload: dict) -> str:
    page_info = payload.get("pageInfo") or {}
    raw_user_context = page_info.get("userContext")
    if not raw_user_context:
        return ""
    try:
        parsed = json.loads(raw_user_context)
    except Exception:
        return ""
    pics = parsed.get("pics") or []
    if pics and isinstance(pics, list):
        first = pics[0] or {}
        return normalize_url(str(first.get("img") or ""))
    return ""


def extract_input_query(payload: dict) -> str:
    page_info = payload.get("pageInfo") or {}
    raw_user_context = page_info.get("userContext")
    if not raw_user_context:
        return ""
    try:
        parsed = json.loads(raw_user_context)
    except Exception:
        return ""
    return str(parsed.get("text") or "").strip()


def normalize_stream(events: list[dict], source_name: str = "") -> dict:

    md_chunks: list[str] = []
    md_seq: int | None = None

    seen_card_keys: set = set()
    seq = 0
    all_text_parts: list[str] = []


    init_seqs: dict[str, int] = {}

    result = {
        "source_name": source_name,
        "input_image_url": "",
        "input_query": "",
        "assistant_name": "购物助手",
        "assistant_icon_url": "",
        "render_trace": "",
        "pvid": "",
        "blocks": [],
        "final_text": "",
        "raw_event_count": len(events),
        "tool_calls": [],
    }

    def flush_markdown() -> None:
        
        nonlocal md_chunks, md_seq
        text = "".join(md_chunks).strip()
        md_chunks = []
        if text and text not in MARKDOWN_EMPTY_SENTINELS:
            result["blocks"].append({
                "seq": md_seq if md_seq is not None else seq,
                "type": "markdown",
                "content": text,
            })
            all_text_parts.append(text)
        md_seq = None

    CARD_TOOL_NAMES = {
        "ImageSameItemsCard",
        "ItemFeedsCard",
        "ItemListCard",
        "MultiRegionImageSameItemsCard",
        "FollowUpQuestionsWithInputCard",
    }

    for wrapped in events:
        payload = wrapped["payload"]
        if not result["input_image_url"]:
            result["input_image_url"] = extract_input_image(payload)
        if not result["input_query"]:
            result["input_query"] = extract_input_query(payload)
        if not result["render_trace"]:
            result["render_trace"] = str(payload.get("render_trace") or "")
        if not result["pvid"]:
            result["pvid"] = str(payload.get("pvid") or "")

        raw_items = payload.get("result") or []
        tool_name = str(payload.get("toolName") or "")
        if tool_name and (not result["tool_calls"] or result["tool_calls"][-1] != tool_name):
            result["tool_calls"].append(tool_name)

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("tItemType") or "") != "nt_title":
                continue
            title_data = item.get("data") or {}
            title = str(title_data.get("title") or "").strip()
            icon_url = normalize_url(str(title_data.get("icon") or "").strip())
            if title:
                result["assistant_name"] = title
            if icon_url:
                result["assistant_icon_url"] = icon_url

        for item in raw_items:
            if item.get("type") == "markdown":
                text = str((item.get("data") or {}).get("text") or "")
                if text and md_seq is None:
                    md_seq = seq
                md_chunks.append(text)



        for item in raw_items:
            if not isinstance(item, dict):
                continue
            inner_data = item.get("data") or {}
            if inner_data.get("renderStatus") == "init":
                item_id = str(item.get("tItemId") or "")
                if item_id:
                    flush_markdown()
                    init_seqs[item_id] = seq

        if tool_name == "ImageSameItemsCard":
            flush_markdown()
            normalized_items = []
            first_result = raw_items[0] if raw_items else {}

            data_field = first_result.get("data") or {}
            items_list = data_field.get("items") or []
            if not items_list:
                trans_items = first_result.get("transItems") or {}
                items_list = trans_items.get("items") or []
            for item in items_list:
                if isinstance(item, dict):
                    normalized_items.append(normalize_product(item))
            card_key = (
                tool_name,
                first_result.get("tItemId"),
                make_items_signature(normalized_items),
            )
            if normalized_items and card_key not in seen_card_keys:
                result["blocks"].append({
                    "seq": seq,
                    "type": "image_same_items_card",
                    "expand_text": str(first_result.get("expandBtnText") or "查看更多同款"),
                    "items": normalized_items,
                })
                seen_card_keys.add(card_key)

        elif tool_name == "ItemListCard":
            flush_markdown()
            normalized_items = []
            first_result = raw_items[0] if raw_items else {}
            data_field = first_result.get("data") or {}
            items_list = data_field.get("items") or []
            if not items_list:

                trans_items = first_result.get("transItems") or {}
                items_list = trans_items.get("items") or []
            if not items_list and len(raw_items) > 1:

                for raw_item in raw_items:
                    if not isinstance(raw_item, dict):
                        continue
                    d = raw_item.get("data") or {}
                    if d.get("itemId") or d.get("item_id") or d.get("id"):
                        items_list.append(d)
                    elif raw_item.get("itemId") or raw_item.get("item_id"):
                        items_list.append(raw_item)
            if not items_list:
                import sys
                print(f"[DEBUG ItemListCard] raw_items empty or no items found. raw_items={raw_items!r}", file=sys.stderr)
            for item in items_list:
                if isinstance(item, dict):
                    normalized_items.append(normalize_product(item))
            card_key = (
                tool_name,
                first_result.get("tItemId"),
                make_items_signature(normalized_items),
            )
            if normalized_items and card_key not in seen_card_keys:
                result["blocks"].append({
                    "seq": seq,
                    "type": "item_list_card",
                    "expand_text": str(first_result.get("expandBtnText") or "查看更多商品"),
                    "items": normalized_items,
                })
                seen_card_keys.add(card_key)

        elif tool_name == "ItemFeedsCard":
            flush_markdown()
            first_result = raw_items[0] if raw_items else {}
            data = first_result.get("data") or {}
            normalized_items = []
            for item in data.get("items") or []:
                if isinstance(item, dict):
                    normalized_items.append(normalize_product(item))
            card_key = (
                tool_name,
                first_result.get("tItemId"),
                make_items_signature(normalized_items),
            )
            if normalized_items and card_key not in seen_card_keys:
                first_show_count = first_result.get("firstShowCount")
                try:
                    first_show_count = max(1, int(float(first_show_count)))
                except Exception:
                    first_show_count = 1
                item_id = str(first_result.get("tItemId") or "")
                card_seq = init_seqs.get(item_id, seq)
                result["blocks"].append({
                    "seq": card_seq,
                    "type": "item_feeds_card",
                    "display_title": str(data.get("cateTitle") or ""),
                    "expand_text": str(first_result.get("expandBtnText") or "查看更多商品"),
                    "first_show_count": first_show_count,
                    "items": normalized_items,
                })
                seen_card_keys.add(card_key)

        elif tool_name == "MultiRegionImageSameItemsCard":
            flush_markdown()
            normalized_items = []
            for item in raw_items:
                data = item.get("data") or {}
                if isinstance(data, dict):
                    normalized_items.append(normalize_product(data))
            card_key = (tool_name, make_items_signature(normalized_items))
            if normalized_items and card_key not in seen_card_keys:
                result["blocks"].append({
                    "seq": seq,
                    "type": "multi_region_image_same_items_card",
                    "items": normalized_items,
                })
                seen_card_keys.add(card_key)

        elif tool_name in {"FollowUpQuestionsWithInputCard", "FollowUpQuestionsCard"}:
            flush_markdown()
            title, queries = extract_follow_up_questions(raw_items)
            card_key = ("follow_up_questions_card", tuple(queries))
            if queries and card_key not in seen_card_keys:
                result["blocks"].append({
                    "seq": seq,
                    "type": "follow_up_questions_card",
                    "title": title,
                    "questions": queries,
                })
                seen_card_keys.add(card_key)

        seq += 1

    flush_markdown()

    result["blocks"].sort(key=lambda block: block["seq"])
    result["final_text"] = "\n\n".join(all_text_parts)
    return result
