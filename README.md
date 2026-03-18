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
