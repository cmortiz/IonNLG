"""Edge-order numbers of Supplementary Note 2 (subsection "Edge order").

Each edge {u, v} was run as the single question (u, v). If the reversed questions win less often than
the measured ones by Delta on average, the win rate of the full 88-question game is omega - 37 Delta / 88.
The test against 86/88 then rejects for every Delta < (88/37) x (leakage tolerance), the tolerance being
the one written by ion_winrates.py. Without an assumption on Delta, the comparison is with 87/88, the
classical value of the 51 circuits as run, and this script gives the one-sided Bernstein p-value against
it (worst-case null variance, as in ion_winrates.py).

Usage: uv run python scripts/edge_order.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ion_winrates as w  # noqa: E402

OMEGA_RUN = 87 / 88  # classical value of the 51 circuits as run

for name in ["Blue", "H2"]:
    p, n, _ = w.SYSTEMS[name]()
    omega = float(w.W @ p)
    tol = w.leakage_tolerance(omega, n)
    edge_loss = 1 - float(p[w.N_VERTEX:].mean())
    print(f"{name}: omega = {omega:.5f}, leakage tolerance = {tol:.5f}, "
          f"Delta < {88 / 37 * tol:.4f} (measured edge loss {edge_loss:.4f}, "
          f"reversed questions may lose up to {edge_loss + 88 / 37 * tol:.4f})")
    if omega > OMEGA_RUN:
        log_p, _ = w.null_log_pvalue(omega, n, OMEGA_RUN)
        print(f"   against 87/88: omega - 87/88 = {omega - OMEGA_RUN:+.5f}, p = {math.exp(log_p):.2e}")
    else:
        print(f"   against 87/88: omega - 87/88 = {omega - OMEGA_RUN:+.5f}, does not exceed")
