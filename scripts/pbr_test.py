"""Run the PBR non-signaling test (Supplementary Note 3) on Blue and H2 with 5-fold CV.

Answers follow the convention of src/IonNLG/adapters/trapped_ion.py: a0 = outcome % 4, a1 = outcome // 4.
Circuit (u, v) gives question x0 = u to player 0 and x1 = v to player 1.
Ratios are smoothed, R_eta = (1 - eta) f_train / P*_NS + eta: a convex mix of the PBR and 1, so the
Bell-like inequality of Tabia et al. still holds and no ratio is zero (an outcome absent from the
training fold no longer sets t = 0). Each fold is reported separately for every eta in ETAS.
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from nonsignaling import NonSignalingRegression, pvalue_score  # noqa: E402
from ion_winrates import RAW, denominator  # noqa: E402

CIRCUITS = RAW / "circuits"
ETAS = [0.01, 0.1, 0.5, 0.9]
pat = re.compile(r"line (\d+) \((\d+), (\d+)\)")
QMAP = {int(m.group(1)): (int(m.group(2)), int(m.group(3)))
        for f in CIRCUITS.glob("*.txt") if (m := pat.match(f.stem))}
assert len(QMAP) == 51 and all(QMAP[i][0] == QMAP[i][1] for i in range(14))


def samples_from_counts(counts_by_circuit, swap=False):
    rows = []
    for i, c in enumerate(counts_by_circuit):
        u, v = QMAP[i]
        for outcome, k in c.items():
            a0, a1 = outcome % 4, outcome // 4
            if swap:
                a0, a1 = a1, a0
            rows.extend([(u, v, a0, a1)] * int(k))
    return np.array(rows)


def blue_counts():
    data = ast.literal_eval((RAW / "Blue data.txt").read_text())
    out = []
    for c in data:
        n = denominator(c.values())
        out.append({x: round(p * n) for x, p in c.items() if p})
    return out


def h2_counts():
    data = json.loads((RAW / "quantinuum_h2_2026-06-15_counts.json").read_text())
    return [{int(b, 2): k for b, k in c.items()} for c in data]


def run(name, counts, swap=False, seed=0):
    X = samples_from_counts(counts, swap)
    P_x = np.zeros((14, 14))
    for i in range(51):
        P_x[QMAP[i]] = 1
    P_x /= P_x.sum()
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    out = []
    for fold, (train, test) in enumerate(kf.split(X)):
        model = NonSignalingRegression(2, 14, 4, P_x=P_x)
        model.fit(X[train])
        r0 = model._R.copy()
        ps = {}
        for eta in ETAS:
            model._R = (1 - eta) * r0 + eta
            ps[eta] = pvalue_score(model.predict(X[test]))
        out.append((model._kl_div, ps))
        print(f"{name}{' (players swapped)' if swap else ''} fold {fold}: N={len(X)} kl={model._kl_div:.4e} "
              + " ".join(f"p_U(eta={e})={p:.3g}" for e, p in ps.items()))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["blue", "h2", "h2-swap"]
    res = {}
    if "blue" in which:
        res["Blue"] = run("Blue", blue_counts())
    if "h2" in which:
        res["H2"] = run("H2", h2_counts())
    if "h2-swap" in which:
        res["H2 swapped"] = run("H2", h2_counts(), swap=True)
    out = Path(__file__).parents[1] / "data/processed_data/trapped_ion/pbr_results.json"
    out.write_text(json.dumps({k: [{"kl": a, "p_U": {str(e): p for e, p in b.items()}} for a, b in v]
                               for k, v in res.items()}, indent=2))
    print("wrote", out)
