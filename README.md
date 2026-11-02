Data and code to support the paper "Nonlocal Games as Cross-Platform Quantum
Benchmarks: Exceeding unconditional classical
bounds on trapped-ion processors" by A. Than et al. (2026).

PNNL Report Number: PNNL-SA-221109

## Structure

- `data/`: contains the raw data from the experiments, as well as the processed data
  used for the figures in the paper. Note that the superconducting data come from 
  [Furches et al. (2025)](https://iopscience.iop.org/article/10.1088/2058-9565/adf1c0) and are
  not included here.
- `src/`: contains the code used to process the data into a common format
- `notebooks/`: Jupyter notebooks used for plotting and PBR protocol analysis
- `scripts/`: analysis scripts for the revised manuscript. `ion_winrates.py` computes the weighted
  win rates and Bernstein bounds for every trapped-ion run and writes
  `data/processed_data/trapped_ion/ion_winrates.json`; `pbr_test.py` runs the PBR non-signaling test on
  the Blue and H2 data and writes `data/processed_data/trapped_ion/pbr_results.json`; `make_figures.py`
  regenerates the win-rate figures and needs the superconducting experiment database of Furches et al.
  (2025), which is not included here.

## Revised manuscript (2026)

The revision adds a Quantinuum H2 run and two IonQ Forte runs, and recomputes the Silver win rate from
the full outcome distributions. Data provenance is in `data/raw_data/trapped_ion/README.md`. To reproduce
the trapped-ion numbers in the paper:

```bash
uv run python scripts/ion_winrates.py
uv run python scripts/pbr_test.py
```
