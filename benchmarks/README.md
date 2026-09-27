# Contrast benchmark

Compare the direct and lookup evaluation modes with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python benchmarks/contrast_lookup.py --output /tmp/contrast_lookup_results.json
```

Use `--sizes`, `--workers` and `--repeats` to change the workload. The JSON output
contains individual timings, medians, window sizes, iteration counts and output
differences. Include measured results and the machine configuration in the pull
request description.
