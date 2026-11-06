"""Do a player's answer frequencies for a question change with the partner's question?

Under no-signaling they should not. For each player and each question asked with at least two partner
questions, this forms the table (partner question x answer) and computes Pearson's chi-square, then sums
over players and questions. Data: exact per-circuit shot counts, the same files as ion_winrates.py.
Answers follow pbr_test.py: a0 = outcome % 4, a1 = outcome // 4; circuit (u, v) asks player 0 u, player 1 v.

The two players' answers are correlated within a circuit, so the summed statistic does not follow the
textbook chi-square law, and shuffling each table separately understates its spread. The p-value therefore
comes from a parametric bootstrap: fit each circuit's joint table to the pooled marginals by iterative
proportional fitting (a non-signaling null, no convex solver), redraw the exact shot counts, recompute.
The script also prints the win rate of that fit next to the data's, Eq. 4 with its 1/88 and 2/88 weights.

Usage: python marginal_homogeneity.py [replicates]
"""
import ast
import json
import re
import sys

import numpy as np

from ion_winrates import RAW, W, denominator

CIRCUITS = RAW / "circuits"
QUESTIONS = {int(m[1]): (int(m[2]), int(m[3])) for f in CIRCUITS.glob("*.txt")
             if (m := re.match(r"line (\d+) \((\d+), (\d+)\)", f.stem))}
assert len(QUESTIONS) == 51


def blue_tables():
    data = ast.literal_eval((RAW / "Blue data.txt").read_text())
    out = np.zeros((51, 4, 4))
    for i, c in enumerate(data):
        n = denominator(c.values())
        for o, p in c.items():
            out[i, o % 4, o // 4] += round(p * n)
    return out


def h2_tables():
    data = json.loads((RAW / "quantinuum_h2_2026-06-15_counts.json").read_text())
    out = np.zeros((51, 4, 4))
    for i, c in enumerate(data):
        for b, k in c.items():
            o = int(b, 2)
            out[i, o % 4, o // 4] += k
    return out


def blocks():
    """(player, question) -> circuits, for questions asked with at least two partner questions."""
    g = {}
    for i, (u, v) in QUESTIONS.items():
        g.setdefault((0, u), []).append(i)
        g.setdefault((1, v), []).append(i)
    return {k: v for k, v in g.items() if len(v) >= 2}


def statistic(tab, blk):
    total, dof = 0.0, 0
    for (player, _), idx in blk.items():
        m = tab[idx].sum(axis=2 - player)  # rows: circuits, columns: this player's answers
        m = m[:, m.sum(0) > 0]
        e = m.sum(1, keepdims=True) * m.sum(0, keepdims=True) / m.sum()
        total += ((m - e) ** 2 / e).sum()
        dof += (m.shape[0] - 1) * (m.shape[1] - 1)
    return total, dof


def non_signaling_null(tab, blk, sweeps=2000):
    """Rescale every joint table until each player's marginal is the same across partner questions."""
    p = tab / tab.sum(axis=(1, 2), keepdims=True)
    n = tab.sum(axis=(1, 2))
    for _ in range(sweeps):
        for (player, _), idx in blk.items():
            axis = 2 - player  # sum over the partner's answers
            marg = p[idx].sum(axis=axis)
            pooled = (marg * n[idx, None]).sum(0) / n[idx].sum()
            scale = np.divide(pooled, marg, out=np.ones_like(marg), where=marg > 0)
            p[idx] = p[idx] * (scale[:, :, None] if player == 0 else scale[:, None, :])
    return p / p.sum(axis=(1, 2), keepdims=True)


def win_rate(tab):
    """Vertex circuits (the first 14) win when the answers match, edge circuits when they differ."""
    same = np.trace(tab / tab.sum(axis=(1, 2), keepdims=True), axis1=1, axis2=2)
    return float(W @ np.r_[same[:14], 1 - same[14:]])


def report(name, tab, reps, seed=20261105):
    blk = blocks()
    chi, dof = statistic(tab, blk)
    null = non_signaling_null(tab, blk)
    n = tab.sum(axis=(1, 2)).astype(int)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for r in range(reps):
        sample = np.stack([rng.multinomial(n[i], null[i].ravel()).reshape(4, 4) for i in range(51)]).astype(float)
        draws[r] = statistic(sample, blk)[0]
    p = (1 + (draws >= chi).sum()) / (reps + 1)
    print(f"{name}: shots {n.min()}-{n.max()} per circuit, chi2 = {chi:.1f} on {dof} dof; "
          f"null mean {draws.mean():.1f}, sd {draws.std():.1f}; bootstrap p = {p:.4f} ({reps} replicates); "
          f"omega {win_rate(tab):.5f} in the data, {win_rate(null):.5f} in the non-signaling fit")


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    report("Blue", blue_tables(), reps)
    report("H2", h2_tables(), reps)
