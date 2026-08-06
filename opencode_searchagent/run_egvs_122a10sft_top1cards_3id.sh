#!/usr/bin/env bash
set -euo pipefail

OPENCODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export API_KEY=""
export AGENTLOOP_MODEL=""
export AGENTLOOP_API_KEY=""
export AGENTLOOP_BASE_URL=""
export real_modelname=""

export AGENTLOOP_PROMPT_FILE="${OPENCODE_ROOT}/prompt/0713_open_searchagent_output_text_with_1card.txt"
export AGENTLOOP_TOOLS="platform_product_search,product_image_search"
export AGENTLOOP_DATASET="${AGENTLOOP_DATASET:-${OPENCODE_ROOT}/data/multirun_hardcase_val/data/multi_run_hardcase_merged_num289_regen.json}"
export AGENTLOOP_CONCURRENCY="4"

export OFFLINE_DETAIL_PATH="${OPENCODE_ROOT}/mmshopbench_ask_ai_benchmark/indexes/image/image_manifest_details.v0.6.unique.jsonl"
export OFFLINE_ITEM_MANIFEST="${OFFLINE_DETAIL_PATH}"
export OFFLINE_TEXT_INDEX_PATH="${OPENCODE_ROOT}/mmshopbench_ask_ai_benchmark/indexes/text/bm25.pkl"
export OFFLINE_TEXT_TOP_K="10"

export JUDGE_MODEL="gpt-4o"
export JUDGE_API_KEY="${API_KEY}"

export AXIS_LABELS="${OPENCODE_ROOT}/mmshopbench_ask_ai_benchmark/axis_labels.jsonl"
export TOPN="20"
export TOPK="4"
export JUDGE_CAP="40"

STEP1_PY="${OPENCODE_ROOT}/examples/run_batch_eval_parallel_mmshopbench_multiturn_hardcase_matched_num57_nothink.py"

export ONLINE_CARD_RENDER_ROOT="${ONLINE_CARD_RENDER_ROOT:-${OPENCODE_ROOT}/../online_card_render}"

STEP1_LOG="$(mktemp)"
trap 'rm -f "$STEP1_LOG"' EXIT

echo "==> [1/5] 运行 agent batch eval ..." >&2
set +e
python3 "$STEP1_PY" 2>&1 | tee "$STEP1_LOG"
STEP1_RC=${PIPESTATUS[0]}
set -e
[[ "${STEP1_RC}" -ne 0 ]] && echo "==> Step1 退出码 ${STEP1_RC}(可能有失败样本),仍尝试继续" >&2

OUTPUT_JSON="$(awk '/^output:/{p=$2} END{print p}' "$STEP1_LOG")"
if [[ -z "${OUTPUT_JSON}" || ! -f "${OUTPUT_JSON}" ]]; then
  echo "ERROR: 未从 Step1 输出解析到有效 output 路径: '${OUTPUT_JSON}'" >&2
  exit 1
fi
OUTPUT_TRACE="${OUTPUT_JSON%.json}-trace.jsonl"
if [[ ! -f "${OUTPUT_TRACE}" ]]; then
  echo "ERROR: 未找到 trace: ${OUTPUT_TRACE}(pool_verify 需要它取候选池)" >&2
  exit 1
fi
echo "==> Step1 output: ${OUTPUT_JSON}" >&2
echo "==> Step1 trace : ${OUTPUT_TRACE}" >&2

echo "==> [2/5] pool_verify + EGVS ..." >&2
export OUTPUT_JSON OUTPUT_TRACE
bash "${OPENCODE_ROOT}/run_pool_verify_egvs.sh"
