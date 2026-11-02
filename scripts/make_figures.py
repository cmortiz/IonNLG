"""Regenerate figures/ion_trap_results_bar.pdf (Fig. 2) and figures/frontier_plot.pdf (Fig. 3).

Trapped-ion values come from data/processed_data/trapped_ion/ion_winrates.json (scripts/ion_winrates.py);
superconducting points are the best run per device from data/db.json, the experiment database of
Furches et al. (2025), which is not included in this repository.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D

ROOT = Path(__file__).parents[1]
FIG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"
R = json.loads((ROOT / "data/processed_data/trapped_ion/ion_winrates.json").read_text())
OMEGA_C = 86 / 88
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

BLUE_HL = "#3AAFB9"   # Blue highlight (as in the published Fig. 2)
H2_HL = "#8C7BC9"     # second system above the bound
GRAY_FACE, GRAY_EDGE = "#BBBBBB", "#888888"
ORANGE = "#F47B20"


def bar_figure():
    order = sorted(
        ["Gold", "Silver", "Aria", "Blue", "Forte (2026-03)", "Forte (2026-06)", "H2"],
        key=lambda k: R[k]["omega"],
    )
    labels = {"Forte (2026-03)": "Forte\n(Mar 2026)", "Forte (2026-06)": "Forte\n(Jun 2026)"}
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=300)
    x = np.arange(len(order))
    for i, k in enumerate(order):
        r = R[k]
        face, edge = GRAY_FACE, GRAY_EDGE
        if k == "Blue":
            face, edge = to_rgba(BLUE_HL, 0.3), BLUE_HL
        elif k == "H2":
            face, edge = to_rgba(H2_HL, 0.3), H2_HL
        ax.bar(i, r["omega"], width=0.6, color=face, edgecolor=edge, linewidth=1.2, zorder=2)
        ax.errorbar(i, r["omega"], yerr=r["ci95"], fmt="none", ecolor=edge, elinewidth=1.2, zorder=3)
        if k == "Silver":  # SPAM-corrected value, dashed outline above the raw bar
            s = R["Silver (SPAM-corrected)"]["omega"]
            ax.bar(i, s - r["omega"], bottom=r["omega"], width=0.6, color="#EEEEEE",
                   edgecolor=GRAY_EDGE, linestyle="--", linewidth=1.0, zorder=1)
            ax.text(i - 0.32, s, "SPAM", fontsize=7, color=GRAY_EDGE, ha="right", va="center")
    # classical bound: orange line with an upward fade
    grad = LinearSegmentedColormap.from_list("fade", [to_rgba(ORANGE, 0.85), to_rgba(ORANGE, 0.0)])
    xlim = (-0.6, len(order) - 0.4)
    ax.imshow(np.linspace(0, 1, 100).reshape(-1, 1), cmap=grad, aspect="auto",
              extent=(*xlim, OMEGA_C, OMEGA_C + 0.006), origin="lower", zorder=0)
    ax.axhline(OMEGA_C, color=ORANGE, linewidth=1.2, zorder=0)
    ax.text(xlim[0] + 0.1, OMEGA_C + 0.0012, r"$\omega_c$", color=ORANGE, fontsize=10, va="bottom")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.92, 1.0)
    ax.set_xticks(x, [labels.get(k, k) for k in order])
    ax.set_xlabel("Device")
    ax.set_ylabel(r"Win rate $\bar{\omega}$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "ion_trap_results_bar.pdf")
    plt.close(fig)


def superconducting_points():
    db = json.loads((ROOT / "data/db.json").read_text())["experiments"].values()
    best = {}
    for e in db:
        d = e["device"]
        if d["type"] != "superconducting":
            continue
        key = (d["provider"], d["name"])
        if key not in best or e["win_rate"]["value"] > best[key]["win_rate"]["value"]:
            best[key] = e
    return [(k[0], e["circuit_data"]["strategy"], e["attributes"]["vertex_win_rate"],
             e["attributes"]["edge_win_rate"]) for k, e in best.items()]


def frontier_figure():
    colors = {"ibm": "#6F5EFE", "rigetti": "#01B5AD", "ionq": "#EE7C37", "duke": "#032169",
              "umd": "#E4173B", "quantinuum": "#2E7D32", "old": "#AAAAAA"}
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=300)
    xlim, ylim = (0.485, 1.0), (0.825, 1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for provider, strategy, wv, we in superconducting_points():
        c = colors["old"] if strategy == "4q" else colors[provider]
        ax.scatter(wv, we, s=55 if strategy == "4q" else 75, color=c, marker="o", zorder=3,
                   edgecolor="none")
    ions = [("Silver", "umd", "Silver"), ("Gold", "duke", "Gold"), ("Blue", "duke", "Blue"),
            ("Aria", "ionq", "Aria"), ("Forte (2026-03)", "ionq", "Forte (Mar)"),
            ("Forte (2026-06)", "ionq", "Forte (Jun)"), ("H2", "quantinuum", "H2")]
    # label anchor (data coords) and alignment; most points get leader lines into the empty
    # region below the diagonal, Forte (Mar) and H2 are labelled beside the marker
    place = {"Blue": ((0.964, 0.954), "left", True), "Silver": ((0.950, 0.940), "left", True),
             "Aria": ((0.937, 0.927), "left", True), "Forte (2026-06)": ((0.924, 0.914), "left", True),
             "Gold": ((0.911, 0.901), "left", True), "Forte (2026-03)": ((0.924, 0.981), "right", False),
             "H2": ((0.985, 0.991), "right", False)}
    for key, prov, label in ions:
        wv, we = R[key]["vertex"], R[key]["edge"]
        ax.scatter(wv, we, s=80, color=colors[prov], marker="^", zorder=4, edgecolor="none")
        (tx, ty), ha, leader = place[key]
        ax.annotate(label, (wv, we), xytext=(tx, ty), textcoords="data", fontsize=6.5, ha=ha,
                    va="center", color="#444444", zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5, shrinkA=1, shrinkB=3)
                    if leader else None)
    # omega = omega_c line: 14 w_v + 74 w_e = 86 (in units of 1/88)
    wv_line = np.array([(86 - 74) / 14, 1.0])
    we_line = (86 - 14 * wv_line) / 74
    ax.plot(wv_line, we_line, color="#DB98F8", linewidth=1.5, zorder=2)
    ax.fill_between(wv_line, we_line, 1, color="#DB98F8", alpha=0.25, zorder=1)

    def screen_angle(x0, y0, x1, y1):
        (a, b), (c, d) = ax.transData.transform([(x0, y0), (x1, y1)])
        return np.degrees(np.arctan2(d - b, c - a))

    xt = 0.913
    ax.text(xt, (86 - 14 * xt) / 74 + 0.0008, "nonlocal", color="#B86FD9", fontsize=6.5,
            rotation=screen_angle(wv_line[0], we_line[0], wv_line[1], we_line[1]),
            rotation_mode="anchor", ha="center", va="bottom")
    ax.plot([0.8, 1.0], [0.8, 1.0], color="#AAAAAA", linewidth=1.3, zorder=1)
    ax.text(0.848, 0.862, r"$\omega_e \geq \omega_v$", color="#AAAAAA", fontsize=8,
            rotation=screen_angle(0.8, 0.8, 1.0, 1.0), rotation_mode="anchor", ha="center", va="bottom")
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel(r"$\omega_v$")
    ax.set_ylabel(r"$\omega_e$")
    handles = [Line2D([], [], marker=m, color=colors[c], linestyle="", markersize=7, label=l)
               for m, c, l in [("o", "old", "Old strategy"), ("o", "ibm", "IBM"), ("o", "rigetti", "Rigetti"),
                               ("^", "ionq", "IonQ"), ("^", "duke", "Duke"), ("^", "umd", "UMD/JQI"),
                               ("^", "quantinuum", "Quantinuum")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=7,
              frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "frontier_plot.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    bar_figure()
    frontier_figure()
    print("wrote", FIG / "ion_trap_results_bar.pdf", "and", FIG / "frontier_plot.pdf")
