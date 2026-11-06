Data and code to support the paper "Nonlocal Games as Cross-Platform Quantum
Benchmarks: Exceeding unconditional classical
bounds on trapped-ion processors" by A. Than et al. (2026).

PNNL Report Number: PNNL-SA-221109

## Structure

- `data/`: contains the raw data from the experiments, as well as the processed data
  used for the figures in the paper. Provenance of the trapped-ion runs (systems, dates, shot counts,
  submission settings) is in `data/raw_data/trapped_ion/README.md`. Note that the superconducting data
  come from [Furches et al. (2025)](https://iopscience.iop.org/article/10.1088/2058-9565/adf1c0) and are
  not included here.
- `src/`: contains the code used to process the data into a common format
- `notebooks/`: Jupyter notebooks used for plotting and PBR protocol analysis. These notebooks and
  `data/processed_data/summary.csv` hold an earlier analysis, and the trapped-ion numbers in the paper
  come from `scripts/`.
- `scripts/`: analysis scripts that produce the trapped-ion numbers in the paper. `ion_winrates.py`
  computes the weighted win rates, Bernstein confidence intervals, p-values, and leakage tolerances for
  every trapped-ion run and writes `data/processed_data/trapped_ion/ion_winrates.json`; `pbr_test.py`
  runs the PBR non-signaling test on the Blue and H2 data and writes
  `data/processed_data/trapped_ion/pbr_results.json`; `edge_order.py` prints the edge-order tolerance and
  the comparison with 87/88, the classical value of the 51 circuits as run (Supplementary Note 2);
  `marginal_homogeneity.py` prints the Pearson test of marginal homogeneity, its bootstrap p-values, and
  the win rate of the fitted non-signaling distribution (Supplementary Note 3);
  `make_figures.py` regenerates the win-rate figures in `figures/` and needs the superconducting
  experiment database of Furches et al. (2025), which is not included here.
- `figures/`: the win-rate figures of the paper (Figs. 2 and 3).

To reproduce the trapped-ion numbers in the paper:

```bash
uv run python scripts/ion_winrates.py
uv run python scripts/pbr_test.py
uv run python scripts/edge_order.py
uv run python scripts/marginal_homogeneity.py
```
