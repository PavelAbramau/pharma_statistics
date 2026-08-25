"""Local EvidenceEvent extraction from adjacent trial-version bodies.

No network — operates entirely on raw/ + history_index. See diff.py for
the extraction rules (ESTIMATED/ACTUAL boundary, posted-date knowability,
directional enrollment/date changes) and extract.py for corpus-wide
orchestration into warehouse.duckdb::evidence_events.
"""
