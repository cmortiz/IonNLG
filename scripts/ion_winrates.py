"""Weighted G14 win rates and Bernstein bounds for every trapped-ion run.

Implements Supplementary Note 2: omega = sum_j w_j p_j with w_j = 1/88 (vertex) and
2/88 (edge); sigma_w^2 = sum_j w_j^2 p_j (1 - p_j) / n_j; Bernstein with C = 1/min_j n_j
(Case 1; with unequal shot counts, 1/min n_j bounds every per-shot summand).
Confidence intervals use the plug-in variance at the measured p_j. The p-value and the leakage
tolerance bound the null hypothesis omega <= omega_0 over all of it: sigma_w^2 is replaced by its
largest value at each null level (water-filling), and the bound is maximised over levels <= omega_0.

Usage: uv run python scripts/ion_winrates.py [raw_dir] [out_json]
"""
import ast
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

RAW = Path(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else Path(__file__).parents[1] / "data/raw_data/trapped_ion"
OUT = Path(sys.argv[2]) if __name__ == "__main__" and len(sys.argv) > 2 else Path(__file__).parents[1] / "data/processed_data/trapped_ion/ion_winrates.json"
N_VERTEX, N_EDGE = 14, 37
W = np.array([1 / 88] * N_VERTEX + [2 / 88] * N_EDGE)
OMEGA_C = 86 / 88
DELTA = 0.05
VERTEX_WIN = {0, 5, 10, 15}  # 4-bit outcomes with Alice's two bits == Bob's two bits


def sigma2_max(level, n):
    """Largest sum_j W_j^2 p_j(1-p_j)/n_j subject to W.p = level: p_j = min(1, 1/2 + mu n_j / (2 W_j))."""
    f = lambda mu: float(W @ np.clip(0.5 + mu * n / (2 * W), 0, 1)) - level
    lo, hi = 0.0, 1.0
    while f(hi) < 0:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    q = np.clip(0.5 + hi * n / (2 * W), 0, 1)
    return float(np.sum(W**2 * q * (1 - q) / n))


def null_log_pvalue(omega, n, level):
    """log of sup over null levels g <= level of the one-sided Bernstein bound at the worst-case variance."""
    c = 1 / n.min()
    grid = np.linspace(level - 0.05, level, 2001)
    vals = [-((omega - g) ** 2 / 2) / (sigma2_max(g, n) + c * (omega - g) / 3) for g in grid]
    return max(vals), float(grid[int(np.argmax(vals))])


def leakage_tolerance(omega, n, alpha=DELTA):
    """Largest delta for which omega <= omega_c + delta is rejected at level alpha."""
    ok = lambda d: omega - OMEGA_C - d > 0 and null_log_pvalue(omega, n, OMEGA_C + d)[0] <= math.log(alpha)
    if not ok(0):
        return None
    lo, hi = 0.0, omega - OMEGA_C
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def bernstein(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    s2 = float(np.sum(W**2 * p * (1 - p) / n))
    c = 1 / n.min()
    omega = float(W @ p)
    L = math.log(2 / DELTA)
    ci = 2 * L * c / 3 + math.sqrt(2 * L * s2)
    eps = omega - OMEGA_C
    pval = math.exp(null_log_pvalue(omega, n, OMEGA_C)[0]) if eps > 0 else None
    return omega, ci, pval


def type_ci(p, n):
    """95% half-width for an equal-weight average over one question type (same Bernstein bound, w_j = 1/k)."""
    p, n = np.asarray(p, float), np.asarray(n, float)
    s2 = float(np.sum(p * (1 - p) / n)) / len(p) ** 2
    L = math.log(2 / DELTA)
    return 2 * L / (3 * n.min()) + math.sqrt(2 * L * s2)


def fmt(value, ci):
    """Value rounded to 3 decimals; uncertainty rounded up in the last digit."""
    unc = math.ceil(round(ci * 1000, 9))
    return f"{value:.3f}({unc})"


def win_from_4bit(probs, i):
    v = sum(probs.get(x, 0.0) for x in VERTEX_WIN) / sum(probs.values())
    return v if i < N_VERTEX else 1 - v


def denominator(values):
    d = 1
    for v in values:
        if v:
            q = Fraction(v).limit_denominator(100000).denominator
            d = d * q // math.gcd(d, q)
    return d


def load_blue():
    data = ast.literal_eval((RAW / "Blue data.txt").read_text())
    p = [win_from_4bit(c, i) for i, c in enumerate(data)]
    n = [denominator(c.values()) for c in data]  # probabilities are k/n_j
    return np.array(p), np.array(n), None


def load_gold():
    data = json.loads((RAW / "Gold data.json").read_text())
    probs = [{int(b, 2): v for b, v in data[str(i)].items()} for i in range(51)]
    p = [win_from_4bit(c, i) for i, c in enumerate(probs)]
    n = [denominator(c.values()) for c in probs]
    return np.array(p), np.array(n), None


def silver_rows(path, nbits):
    rows = []
    for line in path.read_text().strip().splitlines()[1:]:
        parts = line.split('"')
        vals = [float(x) for x in parts[2].strip(",").split(",")]
        assert len(vals) == 2**nbits
        rows.append(vals)
    return rows


def silver_pairing(rows):
    """Find the dropped spare bit and the pairing that makes vertex outcomes agree."""
    best = None
    for drop in range(5):
        keep = [b for b in range(5) if b != drop]
        for a1, a2, b1, b2 in [(keep[0], keep[1], keep[2], keep[3]),
                               (keep[0], keep[2], keep[1], keep[3]),
                               (keep[0], keep[3], keep[1], keep[2])]:
            # vertex win iff bit a1 == bit b1 and bit a2 == bit b2
            def vwin(x, a1=a1, a2=a2, b1=b1, b2=b2):  # bind loop values now
                bit = lambda k: (x >> k) & 1
                return bit(a1) == bit(b1) and bit(a2) == bit(b2)
            vmean = np.mean([sum(r[x] for x in range(32) if vwin(x)) for r in rows[:N_VERTEX]])
            if best is None or vmean > best[0]:
                best = (vmean, vwin)
    return best[1]


def load_silver():
    rows = silver_rows(RAW / "silver_raw_probabilities.csv", 5)
    vwin = silver_pairing(rows)
    p = []
    for i, r in enumerate(rows):
        v = sum(r[x] for x in range(32) if vwin(x)) / sum(r)
        p.append(v if i < N_VERTEX else 1 - v)
    n = [2000] * 28 + [4000] * 23  # see data/raw_data/trapped_ion/README.md
    return np.array(p), np.array(n), None


def load_silver_spam():
    rows = silver_rows(RAW / "silver_spam_corrected_probabilities.csv", 4)
    p = [win_from_4bit({k: v for k, v in enumerate(r)}, i) for i, r in enumerate(rows)]
    n = [2000] * 28 + [4000] * 23
    return np.array(p), np.array(n), None


def load_aria():
    wr = json.loads((RAW / "ionq_winrates.json").read_text())
    return np.array([wr[str(i)] for i in range(51)]), np.full(51, 20000), None


def counts_to_p(counts_list):
    p, n = [], []
    for i, c in enumerate(counts_list):
        tot = sum(c.values())
        v = sum(k for b, k in c.items() if int(b, 2) in VERTEX_WIN) / tot
        p.append(v if i < N_VERTEX else 1 - v)
        n.append(tot)
    return np.array(p), np.array(n)


def load_forte(fname):
    raw = json.loads((RAW / fname).read_text())["results"]
    by_idx = {int(Path(k.replace("\\", "/")).name.split("_")[0]): v for k, v in raw.items()}
    counts = [by_idx[i] for i in range(51)]
    p, n = counts_to_p(counts)
    return p, n, counts


def load_h2():
    counts = json.loads((RAW / "quantinuum_h2_2026-06-15_counts.json").read_text())
    p, n = counts_to_p(counts)
    return p, n, counts


SYSTEMS = {
    "Silver": load_silver,
    "Silver (SPAM-corrected)": load_silver_spam,
    "Gold": load_gold,
    "Blue": load_blue,
    "Aria": load_aria,
    "Forte (2026-03)": lambda: load_forte("ionq_forte_2026-03-26_counts.json"),
    "Forte (2026-06)": lambda: load_forte("ionq_forte_2026-06-05_counts.json"),
    "H2": load_h2,
}

if __name__ == "__main__":
    results = {}
    print(f"{'system':26s} {'omega':>8s} {'ci95':>8s} {'w_v':>7s} {'w_e':>7s} {'n_j':>11s} {'p':>10s}  formatted")
    for name, loader in SYSTEMS.items():
        p, n, _ = loader()
        assert len(p) == 51
        omega, ci, pval = bernstein(p, n)
        wv, we = float(p[:N_VERTEX].mean()), float(p[N_VERTEX:].mean())
        civ, cie = type_ci(p[:N_VERTEX], n[:N_VERTEX]), type_ci(p[N_VERTEX:], n[N_VERTEX:])
        nstr = f"{n.min()}-{n.max()}" if n.min() != n.max() else str(n.min())
        tol = leakage_tolerance(omega, n) if pval is not None else None
        results[name] = dict(omega=omega, ci95=ci, p_value=pval, leakage_tolerance=tol, vertex=wv, edge=we,
                             vertex_ci95=civ, edge_ci95=cie,
                             n_min=int(n.min()), n_max=int(n.max()), n_total=int(n.sum()),
                             formatted=fmt(omega, ci), vertex_formatted=fmt(wv, civ), edge_formatted=fmt(we, cie))
        ps = f"{pval:.2e}" if pval is not None else "-"
        if tol is not None:
            print(f"   {name}: worst-case null level at the sup = {null_log_pvalue(omega, n, OMEGA_C)[1]:.6f} (omega_c = {OMEGA_C:.6f}), leakage tolerance {tol:.5f}")
        print(f"{name:26s} {omega:8.5f} {ci:8.5f} {wv:7.4f} {we:7.4f} {nstr:>11s} {ps:>10s}  "
              f"{fmt(omega, ci)}  {fmt(wv, civ)}  {fmt(we, cie)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")
