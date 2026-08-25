"""One-command launcher for the local labelling app.

    python scripts/run_labelling_app.py [--rebuild] [--port 8420]

--rebuild forces the provisional_programs view to be recomputed from
asset_candidates + raw CT.gov snapshots before serving (otherwise it's
built once automatically the first time it's missing and then reused).
"""
from __future__ import annotations

import argparse
import webbrowser

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rebuild", action="store_true", help="recompute provisional_programs before serving")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.rebuild:
        from pharma_stats.labelling import provisional_programs as pp
        n = pp.materialize()
        print(f"Rebuilt provisional_programs: {n} programs.")

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        webbrowser.open(url)

    uvicorn.run("pharma_stats.labelling.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
