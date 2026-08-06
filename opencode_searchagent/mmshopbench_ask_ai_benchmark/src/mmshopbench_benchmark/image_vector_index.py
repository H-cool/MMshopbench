

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass


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


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        import requests

        resp = requests.get(path_or_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        Path(tmp.name).write_bytes(resp.content)
        return Image.open(tmp.name).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def _resolve_open_clip_source(model_name: str) -> tuple[str, str | None]:
    
    import json as _json

    from open_clip.factory import _MODEL_CONFIGS

    path = Path(model_name)
    config_path = path / "open_clip_config.json"
    if not config_path.exists():

        return model_name, None

    cfg = _json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg.get("model_cfg", cfg)
    preprocess_cfg = cfg.get("preprocess_cfg", {})


    weight_candidates = [
        "open_clip_model.safetensors",
        "open_clip_pytorch_model.bin",
        "model.safetensors",
    ]
    weights = next((str(path / name) for name in weight_candidates if (path / name).exists()), None)
    if weights is None:
        raise FileNotFoundError(
            f"No open_clip weights found in {path} (looked for {weight_candidates})"
        )



    safe = re.sub(r"[^0-9A-Za-z_.-]", "-", str(path.resolve()).strip("/"))
    registered_name = f"local-{safe}"
    _MODEL_CONFIGS[registered_name] = model_cfg

    _resolve_open_clip_source._preprocess_cfg[registered_name] = preprocess_cfg
    return registered_name, weights


_resolve_open_clip_source._preprocess_cfg = {}


def load_model(model_name: str, device: str, trust_remote_code: bool = False):
    
    import torch
    import open_clip

    resolved_name, pretrained = _resolve_open_clip_source(model_name)
    preprocess_cfg = _resolve_open_clip_source._preprocess_cfg.get(resolved_name, {})

    mean = preprocess_cfg.get("mean")
    std = preprocess_cfg.get("std")
    model, _, preprocess = open_clip.create_model_and_transforms(
        resolved_name,
        pretrained=pretrained,
        image_mean=tuple(mean) if mean else None,
        image_std=tuple(std) if std else None,
        image_interpolation=preprocess_cfg.get("interpolation"),
        image_resize_mode=preprocess_cfg.get("resize_mode"),
    )
    model = model.to(device).eval()
    return preprocess, model, torch


def _encode_tensors(model, torch, tensors, device: str) -> np.ndarray:
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.no_grad():
        with torch.autocast(device_type=autocast_device):
            feats = model.encode_image(tensors, normalize=True)
    return feats.detach().float().cpu().numpy()


def encode_images(preprocess, model, torch, images: list[Image.Image], device: str) -> np.ndarray:
    
    tensors = torch.stack([preprocess(image) for image in images]).to(device)
    return _encode_tensors(model, torch, tensors, device)


def encode_images_aligned(
    preprocess, model, torch, images: list[Image.Image], device: str
) -> tuple[np.ndarray, list[int]]:
    

    tensors: list[Any] = []
    ok_indices: list[int] = []
    for idx, image in enumerate(images):
        try:
            tensors.append(preprocess(image))
            ok_indices.append(idx)
        except Exception as exc:
            print(f"skip_preprocess index={idx} error={exc}")
    if not tensors:
        return np.zeros((0, 0), dtype=np.float32), []


    try:
        stacked = torch.stack(tensors).to(device)
        return _encode_tensors(model, torch, stacked, device), ok_indices
    except Exception as exc:
        print(f"batch_encode_failed size={len(tensors)} error={exc}; retrying per-image")

    feats_list: list[np.ndarray] = []
    kept: list[int] = []
    for idx, tensor in zip(ok_indices, tensors):
        try:
            single = torch.stack([tensor]).to(device)
            feats_list.append(_encode_tensors(model, torch, single, device))
            kept.append(idx)
        except Exception as exc:
            print(f"skip_encode index={idx} error={exc}")
    if not feats_list:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(feats_list, axis=0), kept


def save_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def cmd_build(args: argparse.Namespace) -> None:
    rows = [
        row
        for row in iter_jsonl(Path(args.manifest))
        if row.get("image_path") or row.get("image_url")
    ]
    if args.limit:
        rows = rows[: args.limit]

    processor, model, torch = load_model(args.model_name, args.device, args.trust_remote_code)
    embeddings: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images: list[Image.Image] = []
        loaded_rows: list[dict[str, Any]] = []
        for row in batch_rows:
            source = row.get("image_path") or row.get("image_url")
            try:
                images.append(load_image(str(source)))
                loaded_rows.append(row)
            except Exception as exc:
                print(f"skip_image item_id={row.get('item_id')} error={exc}")
        if not images:
            continue


        feats, ok_indices = encode_images_aligned(processor, model, torch, images, args.device)
        if feats.shape[0] == 0:
            continue
        embeddings.append(feats)
        kept_rows.extend(loaded_rows[i] for i in ok_indices)
        print(f"encoded={len(kept_rows)}/{len(rows)}")


    for vector_index, row in enumerate(kept_rows):
        row["vector_index"] = vector_index

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)
    np.save(output_dir / "image_embeddings.npy", matrix.astype(np.float32))
    save_metadata(output_dir / "image_metadata.jsonl", kept_rows)

    print(f"indexed_images={len(kept_rows)}")
    print(f"embedding_shape={matrix.shape}")
    print(f"output_dir={output_dir}")


def cmd_query(args: argparse.Namespace) -> None:
    processor, model, torch = load_model(args.model_name, args.device, args.trust_remote_code)
    matrix = np.load(Path(args.index_dir) / "image_embeddings.npy")
    metadata = list(iter_jsonl(Path(args.index_dir) / "image_metadata.jsonl"))

    if len(metadata) != matrix.shape[0]:
        raise ValueError(
            f"index/metadata mismatch: {matrix.shape[0]} vectors vs {len(metadata)} metadata rows"
        )


    by_index = {int(row["vector_index"]): row for row in metadata if "vector_index" in row}

    query_image = load_image(args.image)
    query_vec = encode_images(processor, model, torch, [query_image], args.device)[0]
    scores = matrix @ query_vec
    top_idx = np.argsort(-scores)[: args.top_k]

    for rank, idx in enumerate(top_idx, 1):
        row = dict(by_index.get(int(idx), metadata[int(idx)]))
        row["rank"] = rank
        row["score"] = round(float(scores[int(idx)]), 6)
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query an open_clip image index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_help = (
        "Local model directory with open_clip_config.json + weights "
        "(e.g. .../marqo-ecommerce-embeddings-L), an hf-hub:... id, "
        "or a built-in open_clip model name."
    )
    default_device = _default_device()

    build = subparsers.add_parser("build")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--model-name", required=True, help=model_help)
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument("--device", default=default_device)
    build.add_argument("--trust-remote-code", action="store_true", help="Accepted for compatibility; unused.")
    build.add_argument("--limit", type=int)
    build.set_defaults(func=cmd_build)

    query = subparsers.add_parser("query")
    query.add_argument("--index-dir", required=True)
    query.add_argument("--image", required=True)
    query.add_argument("--model-name", required=True, help=model_help)
    query.add_argument("--device", default=default_device)
    query.add_argument("--trust-remote-code", action="store_true", help="Accepted for compatibility; unused.")
    query.add_argument("--top-k", type=int, default=10)
    query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
