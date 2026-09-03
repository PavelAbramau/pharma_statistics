"""Financial layer (Product C): synthetic program cost, conviction ratio,
and — as they're built — sponsor filings-derived signals. Everything
here is a *relative* signal for ranking/prediction, never a causal claim
and never presented as exact dollars. See docs/decisions/0003 for the
cost-index construction and its one real leakage risk (site count), and
audit/leakage.md for every time-varying feature's knowability-date
contract.
"""
