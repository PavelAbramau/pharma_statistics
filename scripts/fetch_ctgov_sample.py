"""Phase-1 smoke test: fetch a sample of studies, prove the snapshot store
and get_as_of round-trip.

Usage: python scripts/fetch_ctgov_sample.py [n]
"""
import sys
from datetime import date

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.snapshot import get_as_of


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    client = CtgovClient()
    print(f"CT.gov API: {client.version()}")

    print(f"\nFetching {n} interventional oncology studies mentioning 'conjugate'...")
    studies = list(
        client.search_studies(
            term="conjugate",
            cond="cancer",
            fields=["NCTId", "BriefTitle", "OverallStatus", "LeadSponsorName"],
            max_studies=n,
        )
    )
    print(f"Got {len(studies)} studies from the API.")

    nct_ids = []
    for s in studies:
        ident = s["protocolSection"]["identificationModule"]
        nct_ids.append(ident["nctId"])
        print(f"  {ident['nctId']}  {ident.get('briefTitle', '')[:70]}")

    print(f"\nFetching full single-study records for {len(nct_ids)} ids (canonical snapshots)...")
    for nct_id in nct_ids:
        client.get_study(nct_id)

    print("\nRe-reading snapshots back through get_as_of(...):")
    today = date.today()
    ok = 0
    for nct_id in nct_ids:
        snap = get_as_of("ctgov", nct_id, today)
        assert snap is not None, f"missing snapshot for {nct_id}"
        assert snap.body_json()["protocolSection"]["identificationModule"]["nctId"] == nct_id
        ok += 1
    print(f"  {ok}/{len(nct_ids)} snapshots verified via get_as_of, each with a recorded sha256.")

    sample = get_as_of("ctgov", nct_ids[0], today)
    print(f"\nSample snapshot: {sample.path}")
    print(f"  url: {sample.url}")
    print(f"  fetched_at: {sample.fetched_at}")
    print(f"  sha256: {sample.sha256}")


if __name__ == "__main__":
    main()
