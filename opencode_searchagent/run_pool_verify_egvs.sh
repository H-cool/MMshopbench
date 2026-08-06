#!/usr/bin/env bash
set -euo pipefail

OPENCODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${OPENCODE_ROOT}/mmshopbench_ask_ai_eval_pipline"

: "${OUTPUT_JSON:?请 export OUTPUT_JSON=Step1的output.json}"
: "${OUTPUT_TRACE:=${OUTPUT_JSON%.json}-trace.jsonl}"

OFFLINE_DETAIL_PATH="${OFFLINE_DETAIL_PATH:?请 export OFFLINE_DETAIL_PATH=v0.6详情jsonl}"
AXIS_LABELS="${AXIS_LABELS:-${OPENCODE_ROOT}/mmshopbench_ask_ai_benchmark/axis_labels.jsonl}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
JUDGE_API_KEY="${JUDGE_API_KEY:?请 export JUDGE_API_KEY}"

AGENTLOOP_BASE_URL="${AGENTLOOP_BASE_URL:?请 export AGENTLOOP_BASE_URL}"
AGENTLOOP_API_KEY="${AGENTLOOP_API_KEY:?请 export AGENTLOOP_API_KEY}"
AGENTLOOP_MODEL="${AGENTLOOP_MODEL:?请 export AGENTLOOP_MODEL(self-verify模型)}"

TOPN="${TOPN:-20}"
TOPK="${TOPK:-4}"
JUDGE_CAP="${JUDGE_CAP:-40}"
VERIFY_BATCH_SIZE="${VERIFY_BATCH_SIZE:-5}"
VERIFY_CONCURRENCY="${VERIFY_CONCURRENCY:-4}"
VERIFY_NO_THINK="${VERIFY_NO_THINK:-1}"
VERIFY_TEMPERATURE="${VERIFY_TEMPERATURE:-0.0}"
VERIFY_SELECT_MODE="${VERIFY_SELECT_MODE:-replace}"

export ONLINE_CARD_RENDER_ROOT="${ONLINE_CARD_RENDER_ROOT:-${OPENCODE_ROOT}/../online_card_render}"

RUN_NAME="$(basename "${OUTPUT_JSON%.json}")"
OUT_DIR="${PIPELINE_ROOT}/results/pool_verify/${RUN_NAME}"
JUDGE_DIR="${OUT_DIR}/pool_judge"
mkdir -p "${OUT_DIR}"

echo "==> [1/6] 构造候选池注入版 output.json(topN=${TOPN}) ..." >&2
cd "${PIPELINE_ROOT}"
PYTHONPATH=src python -m mmshopbench_eval.steps.pool_verify_prepare \
  --input-json "${OUTPUT_JSON}" \
  --trace-jsonl "${OUTPUT_TRACE}" \
  --topn "${TOPN}" \
  --out-prefix "${OUT_DIR}/${RUN_NAME}"

INJECTED="${OUT_DIR}/${RUN_NAME}_pool_injected.json"
META="${OUT_DIR}/${RUN_NAME}_meta.jsonl"

echo "==> [2/6] 用原判官对注入池版打分(每卡上限 ${JUDGE_CAP}) ..." >&2
PYTHONPATH=src \
  python -m mmshopbench_eval.steps.judge_multiturn_hardcase_num57_quality_json_test_with_itempv_only_judge_find_item_two_card_offline_judge \
    --input-json "${INJECTED}" \
    --save-prompt-messages \
    --output-dir "${JUDGE_DIR}" \
    --model "${JUDGE_MODEL}" \
    --api-key "${JUDGE_API_KEY}" \
    --max-card-items-for-judge "${JUDGE_CAP}" \
    --max-product-images-per-card "${JUDGE_CAP}" \
    --no-write-dw

echo "==> [3/6] 合并判官结果,算 pool/final/gap(按 A/B/C 三轴) ..." >&2
PYTHONPATH=src python -m mmshopbench_eval.steps.pool_verify_report \
  --judge-results "${JUDGE_DIR}/judge_results.jsonl" \
  --meta "${META}" \
  --axis-labels "${AXIS_LABELS}" \
  --out "${OUT_DIR}/pool_verify.jsonl"

echo "==> [4/6] EGVS self-verify 重选(topK=${TOPK}, batch=${VERIFY_BATCH_SIZE}, concurrency=${VERIFY_CONCURRENCY}, no_think=${VERIFY_NO_THINK}) ..." >&2
NO_THINK_FLAG=""
if [[ "${VERIFY_NO_THINK}" == "1" ]]; then NO_THINK_FLAG="--verify-no-think"; fi
PYTHONPATH=src python -m mmshopbench_eval.steps.egvs_self_verify \
  --pool-verify "${OUT_DIR}/pool_verify.jsonl" \
  --offline-detail "${OFFLINE_DETAIL_PATH}" \
  --mandatory-from "${OUTPUT_JSON}" \
  --base-url "${AGENTLOOP_BASE_URL}" \
  --api-key "${AGENTLOOP_API_KEY}" \
  --verify-model "${AGENTLOOP_MODEL}" \
  --topk "${TOPK}" \
  --concurrency "${VERIFY_CONCURRENCY}" \
  --verify-batch-size "${VERIFY_BATCH_SIZE}" \
  --verify-temperature "${VERIFY_TEMPERATURE}" \
  --select-mode "${VERIFY_SELECT_MODE}" \
  ${NO_THINK_FLAG} \
  --out "${OUT_DIR}/egvs.jsonl"

ORIG_JUDGE_DIR="${OUT_DIR}/orig_card_judge"
if [[ "${SKIP_DIRECT_JUDGE:-0}" == "1" ]]; then
  echo "==> [5/6] 跳过直评 llm_judge(SKIP_DIRECT_JUDGE=1)" >&2
else
  echo "==> [5/6] 直评 llm_judge:对原始卡(未注入)跑 offline judge ..." >&2
  PYTHONPATH=src \
    python -m mmshopbench_eval.steps.judge_multiturn_hardcase_num57_quality_json_test_with_itempv_only_judge_find_item_two_card_offline_judge \
      --input-json "${OUTPUT_JSON}" \
      --output-dir "${ORIG_JUDGE_DIR}" \
      --model "${JUDGE_MODEL}" \
      --api-key "${JUDGE_API_KEY}" \
      --no-write-dw \
    || echo "==> 警告:直评 judge 失败(不影响 egvs 结果)" >&2
fi

DOCS_DIR="$(cd "${OPENCODE_ROOT}/.." && pwd)/docs"
RUN_TS_SHORT="$(echo "${RUN_NAME}" | grep -oE '[0-9]{8}-[0-9]{6}' | head -1)"
[[ -z "${RUN_TS_SHORT}" ]] && RUN_TS_SHORT="${RUN_NAME}"
echo "==> [6/6] 汇总指标 -> ${DOCS_DIR}/egvs_results_log.{md,csv} ..." >&2
python3 "${OPENCODE_ROOT}/egvs_append_summary.py" \
  --output-json "${OUTPUT_JSON}" \
  --egvs-jsonl "${OUT_DIR}/egvs.jsonl" \
  --orig-judge-report "${ORIG_JUDGE_DIR}/judge_report.json" \
  --real-modelname "${real_modelname:-}" \
  --verify-model "${AGENTLOOP_MODEL}" \
  --dataset "${AGENTLOOP_DATASET:-}" \
  --run-ts "${RUN_TS_SHORT}" \
  --summary-md "${DOCS_DIR}/egvs_results_log.md" \
  --summary-csv "${DOCS_DIR}/egvs_results_log.csv" \
  || echo "==> 警告:汇总失败" >&2

echo "==> 完成。结果目录: ${OUT_DIR}" >&2
echo "==> 汇总文件: ${DOCS_DIR}/egvs_results_log.md" >&2
