

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


@dataclass
class SearchHit:
    item_id: str
    score: float
    item: dict[str, Any]


class TextIndex:
    def __init__(self, docs: list[dict[str, Any]], postings: dict[str, dict[int, int]], doc_len: list[int]):
        self.docs = docs
        self.postings = postings
        self.doc_len = doc_len
        self.avgdl = sum(doc_len) / max(1, len(doc_len))

    @classmethod
    def build(cls, items: Iterable[dict[str, Any]]) -> "TextIndex":
        docs: list[dict[str, Any]] = []
        doc_len: list[int] = []
        postings: dict[str, dict[int, int]] = defaultdict(dict)

        for doc_id, item in enumerate(items):
            text = item.get("search_text") or build_search_text(item)
            tokens = tokenize(text)
            if not tokens:
                continue
            compact_item = {
                "item_id": str(item.get("item_id", "")),
                "title": item.get("title", ""),
                "subtitle": item.get("subtitle", ""),
                "category": item.get("category", ""),
                "category_path": item.get("category_path", []),
                "price": item.get("price"),
                "image_url": item.get("image_url", ""),
                "shop_id": str(item.get("shop_id", "")),
                "shop_name": item.get("shop_name", ""),
                "search_text": text,
            }
            docs.append(compact_item)
            doc_len.append(len(tokens))
            for token, freq in Counter(tokens).items():
                postings[token][doc_id] = freq

        return cls(docs=docs, postings=dict(postings), doc_len=doc_len)

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as f:
            pickle.dump(
                {
                    "docs": self.docs,
                    "postings": self.postings,
                    "doc_len": self.doc_len,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "TextIndex":
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(
            docs=payload["docs"],
            postings=payload["postings"],
            doc_len=payload["doc_len"],
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self.docs)
        scores: dict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75

        for token in query_tokens:
            posting = self.postings.get(token)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_id, freq in posting.items():
                denom = freq + k1 * (1 - b + b * self.doc_len[doc_id] / max(self.avgdl, 1e-9))
                scores[doc_id] += idf * (freq * (k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [
            SearchHit(
                item_id=str(self.docs[doc_id].get("item_id", "")),
                score=score,
                item=self.docs[doc_id],
            )
            for doc_id, score in ranked
        ]


def build_search_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "subtitle", "category", "shop_name"):
        value = item.get(key)
        if value:
            parts.append(str(value))
    for value in item.get("category_path") or []:
        if value:
            parts.append(str(value))
    return " ".join(parts)


def cmd_build(args: argparse.Namespace) -> None:
    index = TextIndex.build(iter_jsonl(Path(args.items_jsonl)))
    index.save(Path(args.output))
    print(f"indexed_docs={len(index.docs)}")
    print(f"vocab_size={len(index.postings)}")
    print(f"output={args.output}")


def cmd_query(args: argparse.Namespace) -> None:
    index = TextIndex.load(Path(args.index))
    hits = index.search(args.query, top_k=args.top_k)
    for rank, hit in enumerate(hits, 1):
        item = hit.item
        print(
            json.dumps(
                {
                    "rank": rank,
                    "item_id": hit.item_id,
                    "score": round(hit.score, 6),
                    "title": item.get("title"),
                    "category": item.get("category"),
                    "price": item.get("price"),
                    "image_url": item.get("image_url"),
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query a local text index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--items-jsonl", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.set_defaults(func=cmd_build)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--index", required=True)
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--top-k", type=int, default=10)
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
