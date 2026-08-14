from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from utils.chat_format import build_multi_turn_eval_prompt
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN

from .text import DEFAULT_MAX_LENGTH, head_middle_tail


SAFETY_MARGIN = 128
QUERY_RATIOS = {0: 1.00, 1: 0.65, 5: 0.55, 10: 0.50}


@dataclass
class PackedPrompt:
    prompt_ids: object
    conversations: list[dict[str, str]]
    document_token_ids: list[list[int]]
    document_segments: list[list[tuple[int, int]]]
    graph_rows: list[list[int]]
    prompt_tokens: int
    expanded_graph_tokens: int
    answer_reserve: int
    safety_margin: int
    effective_total: int
    max_length: int
    enable_thinking: bool = False


def _decode(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(map(int, ids)),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _messages(tokenizer, support_ids, support_labels, query_ids, *, has_graph: bool):
    marker = f"{DEFAULT_GRAPH_TOKEN}\n" if has_graph else ""
    messages = []
    for ids, label in zip(support_ids, support_labels):
        messages.append({
            "from": "human",
            "value": (
                f"{marker}Management Discussion and Analysis:\n{_decode(tokenizer, ids)}\n"
                "Classify this company-year as Fraud or Normal. Answer with one word."
            ),
        })
        messages.append({"from": "gpt", "value": "Fraud" if int(label) == 1 else "Normal"})
    messages.append({
        "from": "human",
        "value": (
            f"{marker}Management Discussion and Analysis:\n{_decode(tokenizer, query_ids)}\n"
            "Classify this company-year as Fraud or Normal. Answer with exactly Fraud or Normal."
        ),
    })
    return messages


def _render(tokenizer, messages, *, has_graph: bool):
    return build_multi_turn_eval_prompt(
        tokenizer,
        messages,
        has_graph=has_graph,
        max_length=None,
        chat_template_kwargs={"enable_thinking": False},
    )


def _answer_reserve(tokenizer) -> int:
    return max(
        len(tokenizer(answer, add_special_tokens=False).input_ids)
        for answer in ("Fraud", "Normal")
    )


def _allocate(raw_lengths: list[int], k: int, available: int) -> list[int]:
    if available < 0:
        raise ValueError("fixed template and graph expansion exceed the context window")
    query_ratio = QUERY_RATIOS.get(k)
    if query_ratio is None:
        raise ValueError(f"unsupported K={k}; expected zero/1/5/10")
    support_count = 2 * k
    support_total = available - int(available * query_ratio)
    query_budget = available - support_total
    budgets = [support_total // max(1, support_count)] * support_count
    for index in range(support_total % max(1, support_count)):
        budgets[index] += 1
    budgets.append(query_budget)

    # Water filling: unused short support goes to query, then other supports.
    used = [min(length, budget) for length, budget in zip(raw_lengths, budgets)]
    spare = available - sum(used)
    query_index = len(used) - 1
    take = min(spare, raw_lengths[query_index] - used[query_index])
    used[query_index] += take
    spare -= take
    while spare:
        open_indices = [i for i in range(support_count) if used[i] < raw_lengths[i]]
        if not open_indices:
            break
        changed = False
        share = max(1, spare // len(open_indices))
        for index in open_indices:
            take = min(share, spare, raw_lengths[index] - used[index])
            used[index] += take
            spare -= take
            changed |= take > 0
            if spare == 0:
                break
        if not changed:
            break
    return used


def pack_prompt(
    tokenizer,
    support_texts: Sequence[str],
    support_labels: Sequence[int],
    query_text: str,
    *,
    graph_rows: Sequence[Sequence[int]] | None,
    max_length: int = DEFAULT_MAX_LENGTH,
    safety_margin: int = SAFETY_MARGIN,
) -> PackedPrompt:
    k = len(support_texts) // 2
    if len(support_texts) != 2 * k or len(support_labels) != 2 * k:
        raise ValueError("support must contain K Normal + K Fraud records")
    if k not in QUERY_RATIOS:
        raise ValueError("K must be zero, 1, 5, or 10")
    has_graph = graph_rows is not None
    graph_rows = [list(map(int, row)) for row in (graph_rows or [])]
    if has_graph:
        if len(graph_rows) != 2 * k + 1:
            raise ValueError("one graph row is required for every support and query document")
        if any(len(row) != 32 for row in graph_rows):
            raise ValueError("every FiGraph context must contain exactly 32 node slots")

    raw = [
        list(map(int, tokenizer(str(text), add_special_tokens=False).input_ids))
        for text in [*support_texts, query_text]
    ]
    empty_messages = _messages(tokenizer, [[] for _ in support_texts], support_labels, [], has_graph=has_graph)
    empty_prompt = _render(tokenizer, empty_messages, has_graph=has_graph)
    # A graph marker occupies one tokenizer position and is replaced by 32 projector positions.
    graph_extra = 31 * len(graph_rows)
    answer = _answer_reserve(tokenizer)
    available = max_length - len(empty_prompt) - graph_extra - answer - safety_margin
    budgets = _allocate([len(ids) for ids in raw], k, available)

    def select(current_budgets):
        views = [head_middle_tail(ids, max(1, budget)) for ids, budget in zip(raw, current_budgets)]
        messages = _messages(
            tokenizer,
            [view.token_ids for view in views[:-1]],
            support_labels,
            views[-1].token_ids,
            has_graph=has_graph,
        )
        prompt = _render(tokenizer, messages, has_graph=has_graph)
        total = len(prompt) + graph_extra + answer + safety_margin
        return views, messages, prompt, total

    budgets = [max(1, value) for value in budgets]
    while True:
        views, messages, prompt, total = select(budgets)
        if total <= max_length:
            break
        overflow = total - max_length
        # Supports shrink proportionally first. Query is protected until needed.
        shrinkable = [index for index in range(2 * k) if budgets[index] > 1]
        if not shrinkable and budgets[-1] > 1:
            shrinkable = [len(budgets) - 1]
        if not shrinkable:
            raise ValueError("fixed prompt cannot fit the requested max_length")
        capacity = sum(budgets[index] - 1 for index in shrinkable)
        remove = min(max(overflow, 1), capacity)
        remaining = remove
        for offset, index in enumerate(shrinkable):
            if remaining <= 0:
                break
            slots = len(shrinkable) - offset
            amount = min(budgets[index] - 1, max(1, remaining // slots))
            budgets[index] -= amount
            remaining -= amount

    if total > max_length or not messages or messages[-1]["from"] != "human":
        raise RuntimeError("prompt packing invariant failed")
    return PackedPrompt(
        prompt_ids=prompt,
        conversations=messages,
        document_token_ids=[view.token_ids for view in views],
        document_segments=[list(view.segments) for view in views],
        graph_rows=graph_rows,
        prompt_tokens=len(prompt),
        expanded_graph_tokens=32 * len(graph_rows),
        answer_reserve=answer,
        safety_margin=safety_margin,
        effective_total=total,
        max_length=max_length,
    )


def matched_text_prompt(tokenizer, packed: PackedPrompt, support_labels: Sequence[int]) -> PackedPrompt:
    support = packed.document_token_ids[:-1]
    query = packed.document_token_ids[-1]
    messages = _messages(tokenizer, support, support_labels, query, has_graph=False)
    prompt = _render(tokenizer, messages, has_graph=False)
    answer = _answer_reserve(tokenizer)
    total = len(prompt) + answer + packed.safety_margin
    if total > packed.max_length:
        raise RuntimeError("Text-only Matched unexpectedly exceeds Full GraspLLM context")
    return PackedPrompt(
        prompt_ids=prompt,
        conversations=messages,
        document_token_ids=[list(ids) for ids in packed.document_token_ids],
        document_segments=[list(value) for value in packed.document_segments],
        graph_rows=[],
        prompt_tokens=len(prompt),
        expanded_graph_tokens=0,
        answer_reserve=answer,
        safety_margin=packed.safety_margin,
        effective_total=total,
        max_length=packed.max_length,
    )
