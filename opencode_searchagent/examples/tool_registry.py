from __future__ import annotations

import json
import logging
import os
import threading
from inspect import cleandoc
import sys
from pathlib import Path
from typing import Any

from agentloop import ToolRegistry

logger = logging.getLogger(__name__)


OPEN_ROOT = Path(__file__).resolve().parents[2]
if str(OPEN_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_ROOT))





MMSHOPBENCH_BENCHMARK_ROOT = (
    OPEN_ROOT / "opencode_searchagent" / "mmshopbench_ask_ai_benchmark"
)
MMSHOPBENCH_BENCHMARK_SRC = MMSHOPBENCH_BENCHMARK_ROOT / "src"
if str(MMSHOPBENCH_BENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(MMSHOPBENCH_BENCHMARK_SRC))

import numpy as np

from mmshopbench_benchmark.offline_image_search import (
    get_searcher as get_offline_image_searcher,
)
from mmshopbench_benchmark.image_vector_index import encode_images, load_image
from mmshopbench_benchmark.text_index import TextIndex






OFFLINE_TEXT_INDEX_PATH = Path(
    os.environ.get(
        "OFFLINE_TEXT_INDEX_PATH",
        str(
            MMSHOPBENCH_BENCHMARK_ROOT
            / "indexes"
            / "text"
            / "v0.6_regen_20260721"
            / "bm25.pkl"
        ),
    )
)
_OFFLINE_TEXT_TOP_K = int(os.environ.get("OFFLINE_TEXT_TOP_K", "10"))
_OFFLINE_IMAGE_TOP_K = int(os.environ.get("OFFLINE_IMAGE_TOP_K", "5"))




_OFFLINE_IMAGE_PRE_DEDUP_TOP_K = int(
    os.environ.get("OFFLINE_IMAGE_PRE_DEDUP_TOP_K", "30")
)








_SEARCH_RETURN_FIELD_ALIASES = {"id": "item_id", "image_url": "image"}


def _parse_search_return_fields() -> frozenset[str] | None:
    raw = [
        f.strip().lower()
        for f in os.environ.get("SEARCH_RETURN_FIELDS", "").split(",")
        if f.strip()
    ]
    if not raw:
        return None
    return frozenset(_SEARCH_RETURN_FIELD_ALIASES.get(f, f) for f in raw)


_SEARCH_RETURN_FIELDS = _parse_search_return_fields()


def _field_allowed(canonical: str) -> bool:
    
    return _SEARCH_RETURN_FIELDS is None or canonical in _SEARCH_RETURN_FIELDS







OFFLINE_DETAIL_PATH = Path(
    os.environ.get(
        "OFFLINE_DETAIL_PATH",
        "/path/to/opencode_searchagent/mmshopbench_ask_ai_benchmark/indexes/image/image_manifest_details.v0.5.unique.jsonl",
    )
)

_offline_text_index: TextIndex | None = None
_offline_text_index_lock = threading.Lock()

_offline_detail_index: dict[str, dict[str, Any]] | None = None
_offline_detail_index_lock = threading.Lock()


def get_offline_text_index() -> TextIndex:
    
    global _offline_text_index
    if _offline_text_index is None:
        with _offline_text_index_lock:
            if _offline_text_index is None:
                _offline_text_index = TextIndex.load(OFFLINE_TEXT_INDEX_PATH)
    return _offline_text_index


def _extract_info_fields(row: dict) -> tuple[str, str, str]:
    
    raw = row.get("raw")
    info = raw.get("info") if isinstance(raw, dict) else None

    parsed: dict = {}
    if isinstance(info, dict):
        parsed = info
    elif isinstance(info, str) and info.strip():
        try:
            loaded = json.loads(info)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded

    return (
        str(parsed.get("item_pv_edit", "") or ""),
        str(parsed.get("item_ocr_info", "") or ""),
        str(parsed.get("item_brand_name", "") or ""),
    )


def get_offline_detail_index() -> dict[str, dict[str, Any]]:
    
    global _offline_detail_index
    if _offline_detail_index is None:
        with _offline_detail_index_lock:
            if _offline_detail_index is None:
                detail: dict[str, dict[str, Any]] = {}
                try:
                    with OFFLINE_DETAIL_PATH.open(encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            item_id = str(row.get("item_id", "")).strip()
                            if not item_id:
                                continue
                            pv_edit, ocr_info, brand_name = _extract_info_fields(row)
                            detail[item_id] = {
                                "price": row.get("price"),
                                "item_pv_edit": pv_edit,
                                "item_ocr_info": ocr_info,
                                "item_brand_name": brand_name,


                                "main_image_url": row.get("main_image_url", ""),
                                "sku_image_urls": row.get("sku_image_urls") or [],
                                "detail_image_urls": row.get("detail_image_urls") or [],
                            }
                except OSError as exc:
                    logger.warning(
                        "offline detail index not loaded from %s: %s",
                        OFFLINE_DETAIL_PATH,
                        exc,
                    )
                _offline_detail_index = detail
    return _offline_detail_index

registry = ToolRegistry()

PLATFORM_PRODUCT_SEARCH_DESCRIPTION = cleandoc(
    """
    在购物平台范围内，根据用户或模型生成的查询语义检索符合条件的商品及相关信息；支持搜索平台上其他店铺内的替代品、搭配、配件、补充品推荐及相似款查找等导购场景。
    默认返回基础字段（商品ID、标题、券后价、CPV属性、图文详情、销量、店铺信息等）；当检索语义涉及库存、评价、物流、服务时，通过 field 指定返回的可选字段。
    """
)

PLATFORM_PRODUCT_FIELD_DESCRIPTION = cleandoc(
    """
    当 query 语义提及下述要素时需选择相应可选字段；如不确定，填写 'all'。可多选，去重：
    - 评价：关键词示例（口碑、评价、评论、评分、好评、差评、晒单、问大家、测评）
    - 商品物流：关键词示例（明日达、次日达、当日达、隔日达、发货、配送、物流、包邮、运费、时效、到货时间、快递）
    - 库存：关键词示例（有货、库存、现货、可售量、补货、缺货）
    - 商品服务：关键词示例（退换无忧、保修、官方正品、极速退款、运费险、7天无理由、售后、质保、联保）
    兜底：无法判断或用户明确要求“都看一下”，使用'all'。
    """
)

PLATFORM_PRODUCT_QUERY_DESCRIPTION = cleandoc(
    """
    模型生成的检索查询语义（Q），支持一次提交多个Q进行批量检索。
    每个Q的字符长度一般4-6个词，最多20字符，需要保留核心商品词和修饰词（比如价格、品牌等），不要遗漏核心的用户诉求。
    """
)

PLATFORM_PRODUCT_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "field": {
            "description": PLATFORM_PRODUCT_FIELD_DESCRIPTION,
            "type": "array",
            "items": {
                "description": "可选字段列表，模型可从中选择或使用'all'返回全部可选字段。",
                "type": "string",
                "enum": [
                    "库存",
                    "评价",
                    "商品物流",
                    "商品服务",
                    "all",
                ],
            },
        },
        "query": {
            "description": PLATFORM_PRODUCT_QUERY_DESCRIPTION,
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "query",
    ],
}

PRODUCT_IMAGE_SEARCH_DESCRIPTION = cleandoc(
    """
    **Primary platform visual product search tool.** Searches for visually similar products on the platform
    based on image features. This is the most powerful tool for finding products that look similar to objects
    in the uploaded image (clothing, accessories, furniture, electronics, etc.). Requires specifying a bounding
    box region of the target object to search.
    """
)

PRODUCT_IMAGE_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "img_idx": {
            "description": "The index of the image (starting from 0)",
            "type": "number",
        },
        "region": {
            "description": "Bounding box of the main object to search in the image, formatted as 'x1,y1,x2,y2' (0-1000 relative coordinates).",
            "type": "string",
        },
    },
    "required": [
        "img_idx",
        "region",
    ],
}


def _extract_image_urls(messages) -> list[str]:
    image_urls: list[str] = []
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue

            image_url = part.get("image_url")
            if isinstance(image_url, str) and image_url.strip():
                image_urls.append(image_url.strip())
                continue

            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url.strip():
                    image_urls.append(url.strip())

    return image_urls


def _parse_image_index(value) -> int:
    if isinstance(value, bool):
        raise ValueError("img_idx must be an integer image index")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError("img_idx must be an integer image index")
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError("img_idx must be an integer image index")


def convert_product_image_search(tool_call, context) -> dict[str, str]:
    img_idx = _parse_image_index(tool_call.arguments["img_idx"])
    image_urls = _extract_image_urls(context.messages)

    if not image_urls:
        raise ValueError("product_image_search requires at least one image in message content")
    if img_idx < 0 or img_idx >= len(image_urls):
        raise ValueError(f"img_idx out of range: {img_idx}, available={len(image_urls)}")

    return {
        "image_url": image_urls[img_idx],
        "region": tool_call.arguments["region"],
    }


def _normalize_query_list(query: object) -> list[str]:
    
    if isinstance(query, str):
        candidates = [query]
    elif isinstance(query, list):
        candidates = [item for item in query if isinstance(item, str)]
    else:
        raise ValueError("platform_product_search 'query' must be a string or a list of strings")

    normalized = [item.strip() for item in candidates if item and item.strip()]
    if not normalized:
        raise ValueError("platform_product_search requires a non-empty 'query'")
    return normalized




_TEXT_SEARCH_FIELD_CANON = {
    "id": "item_id",
    "price": "price",
    "image": "image",
    "title": "title",
    "item_pv_edit": "item_pv_edit",
    "item_ocr_info": "item_ocr_info",
    "item_brand_name": "item_brand_name",
}


def platform_product_search(
    query: list[str],
    field: list[str] | None = None,
    tool_call_id: str = "",
) -> str:
    
    index = get_offline_text_index()
    detail_index = get_offline_detail_index()
    merged: dict[str, dict[str, Any]] = {}

    for item in _normalize_query_list(query):
        for hit in index.search(item, top_k=_OFFLINE_TEXT_TOP_K):
            doc = hit.item
            item_id = str(doc.get("item_id", ""))
            if not item_id or item_id in merged:
                continue
            detail = detail_index.get(item_id, {})

            price = detail.get("price")
            if price is None:
                price = doc.get("price")
            full_item = {
                "id": item_id,
                "price": price if price is not None else "",
                "image": doc.get("image_url", ""),
                "title": doc.get("title", ""),
                "item_pv_edit": detail.get("item_pv_edit", ""),
                "item_ocr_info": detail.get("item_ocr_info", ""),
                "item_brand_name": detail.get("item_brand_name", ""),
            }

            merged[item_id] = {
                key: value
                for key, value in full_item.items()
                if _field_allowed(_TEXT_SEARCH_FIELD_CANON[key])
            }

    return json.dumps({"items": list(merged.values())}, ensure_ascii=False)


def convert_platform_product_search(tool_call, context) -> dict[str, object]:
    
    arguments = tool_call.arguments or {}
    kwargs: dict[str, object] = {
        "query": arguments.get("query"),
        "tool_call_id": context.tool_call_id,
    }
    if "field" in arguments:
        kwargs["field"] = arguments.get("field")
    return kwargs


def _format_offline_image_results(results: list[dict[str, Any]]) -> str:
    
    blocks: list[str] = []
    for row in results:
        price = row.get("price")
        price_line = f"price: ¥ {price}" if price not in (None, "") else "price: ¥ "


        line_spec = [
            ("item_id", f"item_id:{row.get('item_id', '')}"),
            ("title", f"title:{row.get('title', '')}"),
            ("price", price_line),
            ("item_pv_edit", f"item_pv_edit:{row.get('item_pv_edit', '')}"),
            ("item_ocr_info", f"item_ocr_info:{row.get('item_ocr_info', '')}"),
            ("item_brand_name", f"item_brand_name:{row.get('item_brand_name', '')}"),
        ]
        blocks.append(
            "\n".join(line for canonical, line in line_spec if _field_allowed(canonical))
        )
    return "\n\n".join(blocks)


def _crop_to_region(image, region: str):
    
    width, height = image.size
    try:
        rel_parts = [float(x) for x in region.split(",")]
        if len(rel_parts) != 4:
            raise ValueError("region must be x1,y1,x2,y2")
        rel_x1, rel_y1, rel_x2, rel_y2 = rel_parts
        x1 = max(0, min(width - 1, int(rel_x1 / 1000.0 * width)))
        y1 = max(0, min(height - 1, int(rel_y1 / 1000.0 * height)))
        x2 = max(0, min(width, int(rel_x2 / 1000.0 * width)))
        y2 = max(0, min(height, int(rel_y2 / 1000.0 * height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid region coordinates after conversion")
    except Exception as exc:
        logger.warning(f"product_image_search: bad region {region!r} ({exc}); using full image")
        return image
    return image.crop((x1, y1, x2, y2))


def product_image_search(image_url: str, region: str) -> str:
    
    searcher = get_offline_image_searcher()
    detail_index = get_offline_detail_index()

    image = load_image(image_url)
    crop = _crop_to_region(image, region)



    query_vec = encode_images(
        searcher.preprocess, searcher.model, searcher.torch, [crop], searcher.device
    )[0]
    scores = searcher.matrix @ query_vec





    candidate_idx = np.argsort(-scores)[:_OFFLINE_IMAGE_PRE_DEDUP_TOP_K]

    results: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    for idx in candidate_idx:
        if len(results) >= _OFFLINE_IMAGE_TOP_K:
            break
        row = dict(searcher.by_index.get(int(idx), searcher.metadata[int(idx)]))
        item_id = str(row.get("item_id", ""))
        if item_id in seen_item_ids:
            continue
        seen_item_ids.add(item_id)
        detail = detail_index.get(item_id, {})
        results.append(
            {
                "rank": len(results) + 1,
                "score": round(float(scores[int(idx)]), 6),
                "item_id": row.get("item_id"),
                "title": row.get("title"),
                "category": row.get("category"),
                "image_url": row.get("image_url"),
                "price": detail.get("price"),
                "item_pv_edit": detail.get("item_pv_edit", ""),
                "item_ocr_info": detail.get("item_ocr_info", ""),
                "item_brand_name": detail.get("item_brand_name", ""),
            }
        )
    return _format_offline_image_results(results)


registry.register(
    name="platform_product_search",
    description=PLATFORM_PRODUCT_SEARCH_DESCRIPTION,
    parameters=PLATFORM_PRODUCT_SEARCH_PARAMETERS,
    func=platform_product_search,
    convert=convert_platform_product_search,
)

registry.register(
    name="product_image_search",
    description=PRODUCT_IMAGE_SEARCH_DESCRIPTION,
    parameters=PRODUCT_IMAGE_SEARCH_PARAMETERS,
    func=product_image_search,
    convert=convert_product_image_search,
)


def build_registry(tool_names, *, source: ToolRegistry | None = None) -> ToolRegistry:
    
    src = source if source is not None else registry
    subset = ToolRegistry()
    for name in tool_names:
        if name not in src.tools:
            raise KeyError(
                f"unknown tool {name!r}; available: {sorted(src.tools)}"
            )
        subset.tools[name] = src.tools[name]
    return subset
