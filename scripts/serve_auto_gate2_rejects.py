"""Serve every program whose CURRENT gold line is an automatic Gate-2
rejection (is_adc=yes, in_scope=no, decided_by=auto — a Layer 1
deterministic scope rule, e.g. heme_only) in a separate, switchable queue
for hand review. Uses latest_by_program, so a program already reopened
and re-decided by a human drops out on its own — nothing here needs to
track that separately.

    python scripts/serve_auto_gate2_rejects.py

Fully reversible: this only reorders/populates session state
(auto_review_order) — it writes no gold. A card here already has a gold
line; the reviewer's answer appends a NEW one that supersedes it
(store.latest_by_program), same override rule as every other review.
"""
from __future__ import annotations

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store


def auto_gate2_reject_ids(gold_records: list[dict]) -> list[str]:
    latest = store.latest_by_program(gold_records)
    return sorted(
        pid for pid, r in latest.items()
        if r.get("decided_by") == "auto" and r.get("gate_reached") == 2
    )


def main() -> None:
    gold_records = store.load_records()
    ids = auto_gate2_reject_ids(gold_records)
    print(f"{len(ids)} program(s) currently auto-decided at Gate 2 (is_adc=yes, in_scope=no, decided_by=auto).")

    programs = pp.load_materialized()
    reviewed = store.reviewed_program_ids(gold_records)
    session = q.load_session()
    if session is None:
        session = q.new_session(programs, exclude_ids=reviewed)

    session["auto_review_order"] = ids
    q.save_session(session)
    print(f"Installed {len(ids)} candidate(s) into the separate auto-review queue.")
    print(f"active_queue is still {session.get('active_queue', 'main')!r} — switch to 'auto_review' "
          "in the app (topbar) whenever you're ready to work through it.")


if __name__ == "__main__":
    main()
