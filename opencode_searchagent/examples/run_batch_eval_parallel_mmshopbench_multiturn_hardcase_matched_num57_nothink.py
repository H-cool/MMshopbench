from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentloop import (
    AgentLoop,
    JsonlRecorder,
    OpenAICompatibleProvider,
    UnifiedMessage,
)
from examples.tool_registry import build_registry









DEFAULT_DATASET = Path(
    "/path/to/opencode_searchagent/data/multirun_hardcase_val/data/multi_run_hardcase_matched_num134.json"
)

DEFAULT_PROMPT = ROOT / "prompt" / "0713_open_searchagent.txt"
DEFAULT_CONCURRENCY = 4



DEFAULT_TOOLS = "product_image_search,platform_product_search"







if os.getenv("AGENTLOOP_SAMPLING_PLAIN", "").strip().lower() in ("1", "true", "yes"):
    DEFAULT_SAMPLING: dict[str, Any] = {}
else:
    DEFAULT_SAMPLING: dict[str, Any] = {
        "extendParams": {
            "thinkingConfig": {
                "includeThoughts": False,
            },
        },
        "extend_fields": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
    }


class _LockedRecorder:
    def __init__(self, recorder: JsonlRecorder) -> None:
        self._recorder = recorder
        self._lock = threading.Lock()

    def record(self, run_record: dict[str, Any]) -> None:
        with self._lock:
            self._recorder.record(run_record)


def main() -> int:
    model = os.getenv("AGENTLOOP_MODEL")
    if not model:
        print("Missing required environment variable: AGENTLOOP_MODEL", file=sys.stderr)
        return 2

    dataset_path = Path(os.getenv("AGENTLOOP_DATASET", str(DEFAULT_DATASET)))
    prompt_path = Path(os.getenv("AGENTLOOP_PROMPT_FILE", str(DEFAULT_PROMPT)))

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    if not prompt_path.exists():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        return 2

    tool_names = [
        name.strip()
        for name in os.getenv("AGENTLOOP_TOOLS", DEFAULT_TOOLS).split(",")
        if name.strip()
    ]
    registry = build_registry(tool_names)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        print("Dataset must be a JSON array", file=sys.stderr)
        return 2



    limit = _limit_from_env()
    if limit is not None:
        samples = samples[:limit]

    concurrency = _concurrency_from_env()

    output_path = _output_path(model=model, dataset_path=dataset_path, prompt_path=prompt_path)
    trace_path = output_path.with_name(output_path.stem + "-trace.jsonl")


    provider = OpenAICompatibleProvider(
        model=model,
        api_key=os.getenv("AGENTLOOP_API_KEY"),
        base_url=os.getenv("AGENTLOOP_BASE_URL", "https://api.openai.com/v1"),
        timeout=float(os.getenv("AGENTLOOP_TIMEOUT", "180")),
        default_sampling=DEFAULT_SAMPLING,
    )
    agent = AgentLoop(
        provider=provider,
        tool_executor=registry,
        recorder=_LockedRecorder(JsonlRecorder(trace_path)),
    )

    total = len(samples)
    print(f"dataset:     {dataset_path}", file=sys.stderr)
    print(f"prompt:      {prompt_path}", file=sys.stderr)
    print(f"output:      {output_path}", file=sys.stderr)
    print(f"trace:       {trace_path}", file=sys.stderr)
    print(f"model:       {model}", file=sys.stderr)
    print(f"tools:       {tool_names}", file=sys.stderr)
    print(f"concurrency: {concurrency}", file=sys.stderr)
    print(f"total samples: {total}", file=sys.stderr)

    succeeded = 0
    failed = 0
    completed = 0



    results: list[dict[str, Any] | None] = [None] * total

    executor = ThreadPoolExecutor(max_workers=concurrency)
    pending: dict[Future[tuple[dict[str, Any], str]], int] = {}

    try:
        for index, sample in enumerate(samples, start=1):
            future = executor.submit(
                _process_sample,
                sample=sample,
                index=index,
                system_prompt=system_prompt,
                agent=agent,
            )
            pending[future] = index

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    record, status = future.result()
                except Exception as exc:

                    record = _build_output(
                        samples[index - 1],
                        final_text="",
                        error=f"worker crashed: {exc}",
                    )
                    status = "failed"

                results[index - 1] = record

                completed += 1
                if status == "succeeded":
                    succeeded += 1
                else:
                    failed += 1

                print(
                    f"[{completed}/{total}] (orig#{index}) "
                    f"sample_id={record.get('sample_id')} status={status}",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print(
            f"interrupted by user; cancelling {len(pending)} pending task(s); "
            "partial results saved.",
            file=sys.stderr,
        )
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        _write_output(output_path, [r for r in results if r is not None])
        return 130
    else:
        executor.shutdown(wait=True)

    _write_output(output_path, [r for r in results if r is not None])

    print(
        f"done. succeeded={succeeded} failed={failed} total={total}",
        file=sys.stderr,
    )
    print(f"output: {output_path}", file=sys.stderr)
    print(f"trace:  {trace_path}", file=sys.stderr)
    return 0 if failed == 0 else 1


def _process_sample(
    *,
    sample: dict[str, Any],
    index: int,
    system_prompt: str,
    agent: AgentLoop,
) -> tuple[dict[str, Any], str]:
    try:
        messages = _build_messages(sample, system_prompt)
    except Exception as exc:
        record = _build_output(sample, final_text="", error=f"build_messages: {exc}")
        return record, "failed"

    started = time.perf_counter()
    try:
        result = agent.run(messages, task_id=str(index))
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        record = _build_output(
            sample,
            final_text="",
            error=str(exc),
            latency_ms=latency_ms,
        )
        return record, "failed"

    latency_ms = (time.perf_counter() - started) * 1000
    record = _build_output(
        sample,
        final_text=result.final_text or "",
        error=result.error,
        latency_ms=latency_ms,
        run_id=result.run_id,
        finish_reason=result.finish_reason,
    )
    status = "failed" if result.error else "succeeded"
    return record, status


def _build_messages(sample: dict[str, Any], system_prompt: str) -> list[UnifiedMessage]:
    
    important = _important_round(sample)

    text_by_round = _round_map(sample.get("query_text"), "value")
    img_by_round = _round_map(sample.get("query_pic_img"), "value")
    output_by_round = _round_map(sample.get("round_results"), "model_output")

    messages: list[UnifiedMessage] = [
        UnifiedMessage(role="system", content=system_prompt)
    ]

    img_counter = 0

    for rnd in range(1, important):
        images = _images_from_query(img_by_round.get(rnd))
        text = _clean_text(text_by_round.get(rnd))
        ai_text = output_by_round.get(rnd) or ""
        human_content, img_counter = _human_content(images, text, img_counter)
        messages.append(UnifiedMessage(role="user", content=human_content))
        messages.append(UnifiedMessage(role="assistant", content=ai_text))

    cur_images = _images_from_query(img_by_round.get(important))
    cur_text = _clean_text(text_by_round.get(important))
    human_content, img_counter = _human_content(cur_images, cur_text, img_counter)
    if not human_content:
        raise ValueError(
            f"important_chat_round {important} produced empty human message"
        )
    messages.append(UnifiedMessage(role="user", content=human_content))

    return messages


def _important_round(sample: dict[str, Any]) -> int:
    
    important = sample.get("important_chat_round")
    if not isinstance(important, int):
        raise ValueError(f"important_chat_round must be an int, got {important!r}")
    if important < 1:
        raise ValueError(f"important_chat_round must be >= 1, got {important}")
    return important


def _round_map(entries: Any, value_key: str) -> dict[int, Any]:
    
    mapping: dict[int, Any] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        rnd = entry.get("chat_round")
        if isinstance(rnd, int):
            mapping[rnd] = entry.get(value_key)
    return mapping


def _images_from_query(query_image: Any) -> list[str]:
    
    if not query_image:
        return []
    if isinstance(query_image, list):
        return [url for url in query_image if url]
    return [query_image]


def _clean_text(query_text: Any) -> str | None:
    
    if not isinstance(query_text, str):
        return None
    text = query_text.strip()
    return text or None


def _human_content(
    images: list[str], text: str | None, counter_start: int
) -> tuple[list[dict[str, Any]], int]:
    
    parts: list[dict[str, Any]] = []
    k = counter_start

    markers: list[str] = []
    for _ in images:
        markers.append(f"[turn=0 pos={k} img_idx={k}]<image>")
        k += 1

    question = f"# 用户输入信息\n## 用户问题\n\n<text>{text}</text>" if text else ""

    if images and text:


        parts.append({"type": "text", "text": question + "".join(markers)})
        for url in images:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    elif images:

        for url, marker in zip(images, markers):
            parts.append({"type": "text", "text": marker})
            parts.append({"type": "image_url", "image_url": {"url": url}})
    elif text:
        parts.append({"type": "text", "text": question})

    return parts, k


def _build_output(
    sample: dict[str, Any],
    *,
    final_text: str,
    error: str | None = None,
    latency_ms: float = 0.0,
    run_id: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    
    record = dict(sample)
    important = sample.get("important_chat_round")
    output_by_round = _round_map(sample.get("round_results"), "model_output")

    record["sample_id"] = sample.get("conversation_id")
    record["chat_round"] = important
    record["ground_truth_output"] = output_by_round.get(important)
    record["pred_model_output"] = final_text
    record["error"] = error
    record["latency_ms"] = latency_ms
    record["run_id"] = run_id
    record["finish_reason"] = finish_reason

    return record


def _write_output(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _output_path(*, model: str, dataset_path: Path, prompt_path: Path) -> Path:


    name_for_file = os.getenv("real_modelname") or model
    model_tag = _sanitize_for_filename(name_for_file)
    dataset_tag = _sanitize_for_filename(dataset_path.stem)
    prompt_tag = _sanitize_for_filename(prompt_path.stem)
    suffix = f"-{model_tag}-{dataset_tag}-{prompt_tag}"

    configured = os.getenv("AGENTLOOP_OUTPUT_PATH")
    if configured:
        base = Path(configured)
        path = base.with_name(f"{base.stem}{suffix}{base.suffix}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = ROOT / "runs" / prompt_tag / f"eval-multiturn-{timestamp}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_for_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "-" for c in value)


def _concurrency_from_env() -> int:
    configured = os.getenv("AGENTLOOP_CONCURRENCY")
    if not configured:
        return DEFAULT_CONCURRENCY
    value = int(configured)
    if value < 1:
        raise ValueError("AGENTLOOP_CONCURRENCY must be greater than 0")
    return value


def _limit_from_env() -> int | None:
    
    configured = os.getenv("AGENTLOOP_LIMIT")
    if not configured:
        return None
    value = int(configured)
    if value < 1:
        raise ValueError("AGENTLOOP_LIMIT must be greater than 0")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
