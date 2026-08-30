"""Silver track: machine-proposed labels, structurally isolated from gold.

Absolute constraint (see store.py): nothing under this package may write
to gold/labels.jsonl, and every record it produces carries labeller="auto".
The gold set is the evaluation set for whatever this package produces, so
it must stay independent of it — see audit/gold_set.py's
"zero auto-sourced records in gold" check for the enforcement side.
"""
