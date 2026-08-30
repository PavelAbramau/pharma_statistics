"""Layer 2: batched model calls on Layer 1's residue only, via the real
Anthropic Message Batches API (50% cost discount, asynchronous).

Batching happens at two levels:
  - PROMPT level: BATCH_SIZE candidates (20) share one prompt/request.
  - JOB level: every k-sample of every 20-candidate group is submitted
    together as custom_id-keyed requests in ONE Anthropic batch job (see
    submit_round), not as separate synchronous calls — this is what
    "using the Batch API" means here, not just batching candidates into a
    prompt.

k=3 adaptive to 5, escalated at the WHOLE 20-CANDIDATE-GROUP level: if any
candidate in a group disagrees across its 3 samples, that group's 2
remaining samples run in a second batch job. Escalating only the
disagreeing candidate (not the whole group) would break prompt batching
for a saving that's small in practice (groups are cheap) — documented
trade-off, not an oversight.

Text evidence (triage/evidence.py) converts a memory question into an
extraction question — the model is told to prefer quoted trial text over
background knowledge; an answer given without supporting text is flagged
from_recall=True and routed toward Layer 3 preferentially (route_to_layer3)
rather than trusted outright — the validation gate (triage/validation.py)
checks empirically whether that trust would have been justified.

Reuses pharma_stats.silver.model_client for the API client, pricing
table, batch helpers, and JSON extraction — a generic Anthropic-API
utility, not a silver-label data path. This module never writes to
silver/labels.jsonl or gold/labels.jsonl.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from pharma_stats.silver import model_client

BATCH_SIZE = 20
INITIAL_K = 3
ESCALATED_K = 5
TEMPERATURE = 1.0
MAX_TOKENS = 4096
PROMPT_VERSION = "layer2-v1"

_SYSTEM = (
    "You are classifying clinical-trial candidates as antibody-drug conjugates (ADCs) — "
    "an antibody or antibody fragment covalently linked to a cytotoxic payload via a "
    "chemical linker. For each candidate, PREFER the trial text evidence given (name, "
    "synonyms, sponsor, conditions studied, trial description/summary text) over your own "
    "background knowledge — quote the exact substring that supports your answer whenever "
    "the evidence states or clearly implies the modality. Only fall back to background "
    "knowledge about the specific named compound when the evidence given does not settle "
    "it — and if you do, set from_recall=true and leave quote null. Require \"unsure\" "
    "whenever you are not confident and no evidence text settles it — do not guess. "
    "Respond with a single JSON array and nothing else."
)


@dataclass
class CandidateAnswer:
    program_id: str
    name: str
    is_adc: str              # "yes" / "no" / "unsure"
    from_recall: bool
    quote: Optional[str]
    k: int
    disagreement: bool
    votes: list = field(default_factory=list)


def group_into_batches(evidences: list[dict], batch_size: int = BATCH_SIZE) -> list[list[dict]]:
    return [evidences[i:i + batch_size] for i in range(0, len(evidences), batch_size)]


def _format_candidate(i: int, ev: dict) -> str:
    lines = [f"{i}. name: {ev['name']}"]
    if ev.get("synonyms"):
        lines.append(f"   synonyms: {', '.join(ev['synonyms'])}")
    if ev.get("lead_sponsor"):
        lines.append(f"   sponsor: {ev['lead_sponsor']}")
    if ev.get("conditions"):
        lines.append(f"   conditions: {', '.join(ev['conditions'])}")
    for snippet in ev.get("text_snippets") or []:
        lines.append(f"   trial text: {snippet!r}")
    return "\n".join(lines)


def build_batch_prompt(group: list[dict]) -> str:
    candidates_text = "\n".join(_format_candidate(i + 1, ev) for i, ev in enumerate(group))
    return f"""Candidates:
{candidates_text}

Respond with a JSON array of exactly {len(group)} objects, one per candidate in the SAME
ORDER as listed above, each shaped exactly:
{{"name": "<candidate name, verbatim from above>", "is_adc": "yes" | "no" | "unsure",
  "from_recall": true | false, "quote": "<exact substring from the trial text relied on, or null>"}}"""


def _round_custom_id(group_idx: int, sample_idx: int) -> str:
    return f"g{group_idx}:s{sample_idx}"


def submit_round(
    groups: list[list[dict]], group_indices: list[int], *, k: int, model: str,
) -> str:
    """Submits k samples of each named group's prompt as ONE Anthropic
    batch job. group_indices lets a second (escalation) round submit for
    only the groups that need it, while custom_id still encodes the
    original group index so results can be re-joined correctly."""
    requests = []
    for group_idx in group_indices:
        prompt = build_batch_prompt(groups[group_idx])
        for sample_idx in range(k):
            requests.append({
                "custom_id": _round_custom_id(group_idx, sample_idx),
                "prompt": prompt, "system": _SYSTEM, "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS, "model": model,
            })
    return model_client.submit_batch(requests, model=model)


def _parse_group_samples(
    group: list[dict], results: dict, group_idx: int, k: int,
) -> tuple[list[Optional[list]], list]:
    """(parsed_samples, usages) for one group's k custom_ids. A missing
    or unparseable sample is None — treated as "no vote", never guessed."""
    parsed, usages = [], []
    for sample_idx in range(k):
        cid = _round_custom_id(group_idx, sample_idx)
        hit = results.get(cid)
        if hit is None:
            parsed.append(None)
            continue
        text, usage = hit
        usages.append(usage)
        parsed.append(model_client.extract_json_array(text))
    return parsed, usages


def _index_by_position_or_name(parsed: list, group: list[dict]) -> dict:
    """{program_id: answer_obj} for one parsed sample. Matches by position
    first (the prompt requires same order); falls back to name matching
    only if the count doesn't match — never silently assumes position
    when the model dropped or reordered an entry."""
    if parsed is not None and len(parsed) == len(group):
        return {ev["program_id"]: ans for ev, ans in zip(group, parsed)}
    by_name = {(a.get("name") or "").strip().lower(): a for a in (parsed or []) if isinstance(a, dict)}
    return {
        ev["program_id"]: by_name[ev["name"].strip().lower()]
        for ev in group if ev["name"].strip().lower() in by_name
    }


def _tally(pid: str, indexed_samples: list[dict]) -> tuple[list[str], list[dict]]:
    answers = [idx[pid] for idx in indexed_samples if pid in idx and isinstance(idx[pid], dict)]
    votes = [a.get("is_adc") for a in answers if a.get("is_adc") in ("yes", "no", "unsure")]
    return votes, answers


def _initial_disagreement(group: list[dict], indexed_samples: list[dict]) -> dict[str, bool]:
    """Per-candidate: did the INITIAL_K round disagree at all? This is the
    escalation trigger and the value CandidateAnswer.disagreement reports
    — a persistent "this one needed a second look" flag, computed ONCE
    from round 1 alone. It deliberately does NOT get recomputed from the
    final (possibly 5-sample) tally: a single round-1 dissenting vote
    survives into that tally forever (majority vote of 5 samples that
    started 2-1 can be at best 4-1, never unanimous), so re-deriving
    "disagreement" from the final tally would flag almost every escalated
    candidate as still disagreeing even when the escalation samples came
    back a clean, confidence-building 2-0 for the majority side."""
    out = {}
    for ev in group:
        votes, _ = _tally(ev["program_id"], indexed_samples)
        out[ev["program_id"]] = (not votes) or (len(set(votes)) > 1)
    return out


def _resolve_group(
    group: list[dict], indexed_samples: list[dict], actual_k: int, initial_disagreement: dict[str, bool],
) -> dict[str, CandidateAnswer]:
    """Final per-candidate answer = MAJORITY (plurality) vote across every
    sample actually drawn (3 or, after escalation, 5) — per the explicit
    spec ("k=3 adaptive: ... majority vote per candidate"), not the
    strict-unanimity abstention rule silver/sampling.py uses elsewhere.
    disagreement on the returned CandidateAnswer is the INITIAL_K-round
    flag (see _initial_disagreement), not re-derived here — that's what
    route_to_layer3 uses to preferentially re-check anything that needed
    a second look, independent of whether the majority vote itself later
    looked confident."""
    out = {}
    for ev in group:
        pid = ev["program_id"]
        votes, answers = _tally(pid, indexed_samples)
        disagreement = initial_disagreement.get(pid, True)
        if not votes:
            out[pid] = CandidateAnswer(pid, ev["name"], "unsure", False, None, actual_k, True, [])
            continue
        top_value, _top_count = Counter(votes).most_common(1)[0]
        winning = [a for a in answers if a.get("is_adc") == top_value]
        from_recall = bool(winning) and sum(1 for a in winning if a.get("from_recall")) > len(winning) / 2
        quote = next((a.get("quote") for a in winning if a.get("quote")), None)
        out[pid] = CandidateAnswer(
            program_id=pid, name=ev["name"], is_adc=top_value,
            from_recall=from_recall, quote=quote, k=actual_k, disagreement=disagreement, votes=votes,
        )
    return out


def run_layer2(
    evidences: list[dict], *, model: str = model_client.DEFAULT_MODEL,
) -> tuple[dict[str, CandidateAnswer], dict]:
    """Full Layer 2 pass: group into 20-candidate batches, submit k=3 for
    every group in ONE batch job, poll, escalate to k=5 for whichever
    groups had any internal disagreement (a second, smaller batch job),
    resolve per-candidate answers. Returns ({program_id: CandidateAnswer},
    log) — log carries per-group k/escalation/cost for the full-reasoning
    record, same spirit as silver/prompts.py's per-question log."""
    groups = group_into_batches(evidences)
    all_indices = list(range(len(groups)))

    round1_batch_id = submit_round(groups, all_indices, k=INITIAL_K, model=model)
    model_client.poll_batch_until_done(round1_batch_id)
    round1_results = model_client.collect_batch_results(round1_batch_id, model=model)

    group_samples: dict[int, list[dict]] = {}
    group_usages: dict[int, list] = {}
    group_initial_disagreement: dict[int, dict[str, bool]] = {}
    escalate_indices = []
    for group_idx in all_indices:
        parsed, usages = _parse_group_samples(groups[group_idx], round1_results, group_idx, INITIAL_K)
        indexed = [_index_by_position_or_name(p, groups[group_idx]) for p in parsed]
        group_samples[group_idx] = indexed
        group_usages[group_idx] = usages
        initial_disagreement = _initial_disagreement(groups[group_idx], indexed)
        group_initial_disagreement[group_idx] = initial_disagreement
        if any(initial_disagreement.values()):
            escalate_indices.append(group_idx)

    if escalate_indices and ESCALATED_K > INITIAL_K:
        round2_batch_id = submit_round(groups, escalate_indices, k=ESCALATED_K - INITIAL_K, model=model)
        model_client.poll_batch_until_done(round2_batch_id)
        round2_results = model_client.collect_batch_results(round2_batch_id, model=model)
        for group_idx in escalate_indices:
            # round-2 custom_ids reuse sample_idx 0..(k-initial_k-1); shift
            # so they land after round 1's samples for this same group
            extra_parsed = []
            for sample_idx in range(ESCALATED_K - INITIAL_K):
                cid = _round_custom_id(group_idx, sample_idx)
                hit = round2_results.get(cid)
                if hit is None:
                    extra_parsed.append(None)
                    continue
                text, usage = hit
                group_usages[group_idx].append(usage)
                extra_parsed.append(model_client.extract_json_array(text))
            group_samples[group_idx] += [
                _index_by_position_or_name(p, groups[group_idx]) for p in extra_parsed
            ]

    results: dict[str, CandidateAnswer] = {}
    per_group_log = []
    total_usages = []
    for group_idx in all_indices:
        actual_k = len(group_samples[group_idx])
        resolved = _resolve_group(
            groups[group_idx], group_samples[group_idx], actual_k, group_initial_disagreement[group_idx],
        )
        results.update(resolved)
        total_usages += group_usages[group_idx]
        per_group_log.append({
            "group_idx": group_idx, "size": len(groups[group_idx]),
            "actual_k": actual_k, "escalated": group_idx in escalate_indices,
        })

    log = {
        "prompt_version": PROMPT_VERSION, "model": model, "n_groups": len(groups),
        "n_escalated_groups": len(escalate_indices), "per_group": per_group_log,
        "usage": {
            "calls": len(total_usages), "input_tokens": sum(u.input_tokens for u in total_usages),
            "output_tokens": sum(u.output_tokens for u in total_usages),
            "cost_usd": sum(model_client.estimate_cost(u.input_tokens, u.output_tokens, u.model, batch=True)
                             for u in total_usages),
        },
    }
    return results, log


def route_to_layer3(answer: CandidateAnswer) -> bool:
    """Layer 3 (web search) is reached preferentially by: outright
    "unsure", persistent disagreement even at escalated_k, or an answer
    whose winning vote only ever came from background knowledge
    (from_recall) — see triage/validation.py, which checks empirically
    whether from_recall answers are trustworthy at all before this
    default is ever relaxed."""
    return answer.is_adc == "unsure" or answer.disagreement or answer.from_recall
