

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from mmshopbench_benchmark.image_vector_index import (
    encode_images,
    iter_jsonl,
    load_image,
    load_model,
)





_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDEX_DIR = os.environ.get(
    "OFFLINE_IMAGE_INDEX_DIR",
    str(_REPO_ROOT / "indexes" / "image" / "vector_0722"),
)
DEFAULT_MODEL_NAME = os.environ.get(
    "OFFLINE_IMAGE_MODEL_NAME",
    "/path/to/marqo-ecommerce-embeddings-L",
)
DEFAULT_TOP_K = int(os.environ.get("OFFLINE_IMAGE_TOP_K", "5"))


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_DEVICE = os.environ.get("OFFLINE_IMAGE_DEVICE", _default_device())


class OfflineImageSearcher:
    

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.device = device


        self.matrix = np.load(self.index_dir / "image_embeddings.npy")
        self.metadata = list(iter_jsonl(self.index_dir / "image_metadata.jsonl"))



        for _row in self.metadata:
            if not _row.get("item_id") and _row.get("item_ids"):
                _row["item_id"] = str(_row["item_ids"][0])
            if not _row.get("title") and _row.get("titles"):
                _row["title"] = _row["titles"][0]
            if not _row.get("category") and _row.get("categories"):
                _row["category"] = _row["categories"][0]
        if len(self.metadata) != self.matrix.shape[0]:
            raise ValueError(
                f"index/metadata mismatch: {self.matrix.shape[0]} vectors "
                f"vs {len(self.metadata)} metadata rows"
            )


        self.by_index = {
            int(row["vector_index"]): row
            for row in self.metadata
            if "vector_index" in row
        }


        self.preprocess, self.model, self.torch = load_model(self.model_name, self.device)

    def search_by_image(self, path_or_url: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        
        query_image = load_image(path_or_url)


        query_vec = encode_images(
            self.preprocess, self.model, self.torch, [query_image], self.device
        )[0]
        scores = self.matrix @ query_vec
        top_idx = np.argsort(-scores)[:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_idx, 1):
            row = dict(self.by_index.get(int(idx), self.metadata[int(idx)]))
            results.append(
                {
                    "rank": rank,
                    "score": round(float(scores[int(idx)]), 6),
                    "item_id": row.get("item_id"),
                    "title": row.get("title"),
                    "category": row.get("category"),
                    "image_url": row.get("image_url"),
                }
            )
        return results





_searcher: OfflineImageSearcher | None = None
_searcher_lock = threading.Lock()


def get_searcher() -> OfflineImageSearcher:
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                _searcher = OfflineImageSearcher()
    return _searcher


def image_search(image_urls: list[str], top_k: int = DEFAULT_TOP_K) -> str:
    
    searcher = get_searcher()
    merged: list[dict[str, Any]] = []
    for image_url in image_urls:
        merged.extend(searcher.search_by_image(image_url, top_k=top_k))
    return json.dumps(merged, ensure_ascii=False)
