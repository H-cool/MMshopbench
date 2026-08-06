# Search Agent — EGVS Evaluation Minimal Reproduction Package

This directory is the minimal reproducible code package for the paper. It contains only the code required by the entry script
`opencode_searchagent/run_egvs_122a10sft_top1cards_3id.sh`, plus the smaller data files.
The large indexes / models are not included and must be placed as described below.

The entry script does two things:
1. **Step1 — agent batch eval**: runs multi-turn hardcase evaluation using the offline
   retrieval tools (`platform_product_search` / `product_image_search`), producing
   `output.json` + `-trace.jsonl`.
2. **pool_verify + EGVS**: inject the candidate pool → re-judge with the original judge →
   compute the pool/final/gap three axes → EGVS self-verification & reselection → direct
   scoring → aggregate into `docs/egvs_results_log.{md,csv}`.

## Directory structure

```
opencode_searchagent/
  run_egvs_122a10sft_top1cards_3id.sh   # entry (one-click script)
  run_pool_verify_egvs.sh                   # pool_verify + EGVS sub-pipeline
  egvs_append_summary.py                    # metrics aggregation
  agentloop/                                # minimal agent loop (pure standard library)
  examples/
    run_batch_eval_parallel_...num57_nothink.py   # Step1 entry
    tool_registry.py                        # keeps only the 2 offline tools used
  mmshopbench_ask_ai_benchmark/src/mmshopbench_benchmark/
    text_index.py           # BM25 text retrieval (standard library)
    image_vector_index.py   # open_clip image encoding
    offline_image_search.py # image vector retrieval
  mmshopbench_ask_ai_eval_pipline/src/mmshopbench_eval/
    config.py partition.py multiturn_benchmark.py
    rendering/stream_normalizer.py
    steps/{pool_verify_common,pool_verify_prepare,pool_verify_report,
           egvs_self_verify,score_item_id_recall,
           judge_..._offline_judge}.py
  prompt/0713_open_searchagent_output_text_with_1card.txt   # included
  data/multirun_hardcase_val/data/multi_run_hardcase_merged_num289_regen.json  # included
  mmshopbench_ask_ai_benchmark/axis_labels.jsonl              # included
docs/                                       # metrics aggregation output directory
```

## Large files you must place yourself (not included in this package)

Due to the upload size limit for supplementary material, the large files below (indexes and
models, ~9 GB in total) are not shipped with the package — only the code and small data are
kept. Place them at the corresponding paths (or point to existing locations via environment
variables) as listed in the table, and you are ready to run.

| File | Size | Placement path / environment variable |
| --- | --- | --- |
| BM25 text index `bm25.pkl` | ~87 MB | `mmshopbench_ask_ai_benchmark/indexes/text/bm25.pkl` (`OFFLINE_TEXT_INDEX_PATH`) |
| Product detail evidence `image_manifest_details.v0.6.unique.jsonl` | ~1.14 GB | `mmshopbench_ask_ai_benchmark/indexes/image/…` (`OFFLINE_DETAIL_PATH`) |
| Image vectors `image_embeddings.npy` | ~424 MB | `mmshopbench_ask_ai_benchmark/indexes/image/vector_0722/` (`OFFLINE_IMAGE_INDEX_DIR`) |
| Image metadata `image_metadata.jsonl` | ~45 MB | same directory as above |
| Marqo-Ecommerce CLIP model | ~7.6 GB | any path, specified via `OFFLINE_IMAGE_MODEL_NAME` (default `/path/to/marqo-ecommerce-embeddings-L`) |

Each empty `indexes/**` directory contains a `PUT_DATA_HERE.txt` note.

## Python dependencies

- Standard library + `numpy`, `pillow` (PIL), `requests`
- `product_image_search` image search additionally requires `torch`, `open_clip_torch`
  (optional `pillow-heif` for heic support)
- No need to install this package: each script locates the source via `PYTHONPATH` / `sys.path`

## Run

```bash
cd opencode_searchagent
bash run_egvs_122a10sft_top1cards_3id.sh
```

Before running, you must configure the model endpoint, API key, concurrency, etc. yourself:
fill in the configuration block at the top of the script
(`AGENTLOOP_MODEL` / `AGENTLOOP_BASE_URL` / `AGENTLOOP_API_KEY` / `JUDGE_MODEL` / `JUDGE_API_KEY`)
and the actual paths for the large files in the table above.
