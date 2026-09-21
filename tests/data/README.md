# Solver reference fixture

`solver_reference.npz` contains the observations and single-thread QuantLet
reference components from `mkuziuk/imf` commit
`0ba4c72`, `experiments/solver-speed/results/`:

- `your_gd_1000_example.npz`: H=0.4, squared-triangular kernel, the original
  integer-window schedule, seed 777.
- `quantlet_zero_2000_example.npz`: H=1, classical Epanechnikov kernel,
  h1=0.2, a=sqrt(2), eight bandwidths, zero replacement, seed 2026.

The arrays are copied without modification. Only unused clean signals, time
grids and lookup-GD outputs are omitted. The scalar reference uses the QuantLet
solver recorded at commit `58d6762cccbdce3abe5e183ceaf9564bf46768ca` in the
research benchmark metadata. The benchmark checks convergence and agreement
before saving its reference outputs.

These are tolerance-based cross-solver fixtures. The existing regression
tests separately preserve the direct-erf notebook's numerical values.
