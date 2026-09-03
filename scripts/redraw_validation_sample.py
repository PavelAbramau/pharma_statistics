"""Redraw the 80-sample Layer 2/3 validation gate against CURRENT staged
decisions and install it as the separate validation queue (session
active_queue stays "main" — this only makes the validation queue
available to switch to, it never switches to it).

Tries the original design first (text-grounded/recall-grounded x
accept/reject, 20/cell) and reports exactly why it fails before falling
back to the source x direction substitute — same two-step process as
the very first draw, repeated here because "redraw" means redo it
properly, not just accept the substitute silently.

    python scripts/redraw_validation_sample.py

Overwrites data/triage_validation_sample.json. Safe to run any time —
drawing against a larger staged-decision pool only improves feasibility,
never breaks an in-progress review (compute_agreement operates on
whatever sample is saved at draw time; new records staged afterward
don't affect a sample already drawn).
"""
from __future__ import annotations

from pharma_stats.config import REPORTS_DIR
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store
from pharma_stats.triage import report as trep
from pharma_stats.triage import staging
from pharma_stats.triage import validation as tval


def main() -> None:
    staged = staging.latest_by_program(staging.load_records())
    # decided_layer23_only excludes Layer 1/1.5 (apply.py stages those too,
    # for the audit trail, even though they commit straight to gold) —
    # assert_no_layer1 requires the CALLER to have already done this, not
    # just trust the draw functions' own internal filtering.
    decisions = tval.decided_layer23_only(list(staged.values()))
    print(f"{len(decisions)} Layer 2/3 decisions on file (Layer 1/1.5 excluded).")

    try:
        sample = tval.draw_stratified_sample(decisions, strict=True)
        design = "original (text/recall x accept/reject)"
    except tval.StratificationError as e:
        print(f"Original design infeasible: {e}\n")
        sample, info = tval.draw_substitute_sample(decisions)
        design = f"substitute (source x direction): {info['main_cells']}, " \
                 f"appendix (no_usable_evidence) n={info['appendix_count']}"

    tval.assert_no_layer1(decisions)
    tval.save_validation_sample(sample)
    print(f"Drew {len(sample)} candidate(s) using the {design} design -> {tval.VALIDATION_SAMPLE_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    blind = REPORTS_DIR / "triage_validation_blind.html"
    blind.write_text(trep.render_validation_blind_html(sample), encoding="utf-8")
    print(f"Blind review page: {blind}")

    programs = pp.load_materialized()
    gold_records = store.load_records()
    reviewed = store.reviewed_program_ids(gold_records)
    sample_ids = [d["program_id"] for d in sample]
    already_reviewed = [pid for pid in sample_ids if pid in reviewed]
    to_serve = [pid for pid in sample_ids if pid not in reviewed]

    session = q.load_session()
    if session is None:
        session = q.new_session(programs, exclude_ids=reviewed)
    session["validation_order"] = to_serve
    q.save_session(session)

    print(f"\nInstalled {len(to_serve)} candidate(s) into the separate validation queue "
          f"({len(already_reviewed)} of the {len(sample)} drawn were already reviewed since the draw).")
    print(f"active_queue is still {session.get('active_queue', 'main')!r} — main queue keeps serving. "
          "Switch to the validation queue in the app (topbar) whenever you're ready to judge it.")


if __name__ == "__main__":
    main()
