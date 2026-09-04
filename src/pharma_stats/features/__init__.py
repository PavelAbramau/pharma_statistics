"""Program x month feature panel — the input to models/. Every
time-varying column is resolved through the same time-cut-safe
versioned-history path finance/cost_model.py already established (never
a current-state read for anything but the static universe-membership
properties docs/decisions/0001 whitelists). See knowability.py for the
per-feature knowability-date registry and as_of_probe.py for the
regression check that enforces it.
"""
