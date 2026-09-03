"""Serve the drawn validation sample blind, through the normal labelling
app — no new UI. The app already never shows a triage verdict for
anything (triage/staging.py is invisible to app.py/triage_serve.py except
for the committable Layer 1/1.5 rejections that get dropped from the
queue entirely), so pointing the session's queue at exactly these
program_ids, in order, IS a blind serve: the reviewer sees the same
card they'd see for any other program, judges normally through the same
three gates, and their real gate-reached decision lands in
gold/labels.jsonl exactly like any other review — which is also what
lets triage/validation.compute_agreement compare it against the staged
verdict afterward.

    python scripts/serve_validation_sample.py

Idempotent: removes the sample ids from wherever they currently sit in
the queue and reinserts them at the front, so re-running this after a
partial session just re-prioritizes whatever's left; it never duplicates
an id or drops the rest of the queue.
"""
from __future__ import annotations

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store as gold_store
from pharma_stats.triage import validation as val


def main() -> None:
    sample = val.load_validation_sample()
    if not sample:
        print(f"No validation sample at {val.VALIDATION_SAMPLE_PATH} — nothing to serve.")
        return
    sample_ids = [d["program_id"] for d in sample]

    reviewed = gold_store.reviewed_program_ids(gold_store.load_records())
    already_reviewed = [pid for pid in sample_ids if pid in reviewed]
    to_serve = [pid for pid in sample_ids if pid not in reviewed]
    if already_reviewed:
        print(f"{len(already_reviewed)} sample id(s) already reviewed since the draw — skipping those, "
              f"agreement will still compare them from gold as-is.")

    session = q.load_session()
    if session is None:
        programs = pp.load_materialized()
        session = q.new_session(programs, exclude_ids=reviewed)
        print("No existing session — created a fresh one.")

    session["order"] = to_serve + [pid for pid in session["order"] if pid not in sample_ids]
    q.save_session(session)
    print(f"Prioritized {len(to_serve)} validation-sample candidate(s) at the front of the queue "
          f"({len(sample)} drawn total, {len(already_reviewed)} already done). "
          "They'll come up next in the normal app — same gates, same blind rules, no verdict shown.")


if __name__ == "__main__":
    main()
