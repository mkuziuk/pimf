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

## Newton benchmark

Compare the solvers on the bursty signal from issue #3:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python benchmarks/newton.py --output /tmp/newton_results.json
```

The script uses one worker and the full schedule from `h1=0.25` down to
`h_min=1.5/L`, with `a=sqrt(2)`. It compares GD at `max_iter=60, tol=1e-6`
with Newton at `max_iter=100, tol=1e-12`. After one warmup, three runs alternate
solver order. Input generation and imports are outside the timed region.
JSON output records all timings, environment details and Newton stage diagnostics.
GD reuses the score at its last evaluated iterate, before the final update.
Newton includes score probes that verify positional accuracy before accepting
a small step, and local probes that tighten wide brackets before bisection.
Both solvers use `on_unconverged="ignore"` to keep warnings for intentionally
capped GD stages out of the timed region; convergence diagnostics are retained.

Measured on an Apple M4 Max, macOS 26.6.2, Python 3.13.13 and NumPy 2.5.1,
with the thread limits above, median wall times were:

- L=1024: GD 0.4355 s, Newton 0.0989 s, a 4.40x speedup.
- L=2048: GD 1.6376 s, Newton 0.3213 s, a 5.10x speedup.
- L=4096: GD 6.9130 s, Newton 1.3803 s, a 5.01x speedup.

Every Newton window converged in at most six iterations, with no bisection
fallbacks on these three signals. Their largest absolute returned scores
were at most 1.13e-15. Default GD capped 7 of 15, 9 of 17 and 11 of 19 stages,
respectively. At L=2048, its component-and-residual stack differed from
Newton by 14.04% in relative L2, reproducing issue #3.

The separate spike window from issue #3 reaches 2.682593636676784 in seven
iterations with one bisection fallback. Across all 64 windows of that stage,
there are 7 fallbacks and no unfinished windows. These figures describe this
workload and machine; compare timings on your own signals.

## Starting-point comparison

The round-2 comparison used the same machine and thread limits as above.
For each start, one warmup preceded three measured runs at each bursty signal
size; candidate order reversed on alternate rounds. Times below are medians
for a full decomposition. All candidates except the original use the repaired
bracket probes and constant-window handling.

| Start | L=1024, s | L=2048, s | L=4096, s | Unfinished outlier windows |
| --- | ---: | ---: | ---: | ---: |
| Original weighted mean | 0.1051 | 0.3459 | 1.3657 | 7605 |
| Weighted mean with repaired brackets | 0.0986 | 0.3269 | 1.3546 | 7605 |
| Unweighted median | 0.1138 | 0.4042 | 1.5640 | 0 |
| Weighted median | 0.1464 | 0.5669 | 2.3738 | 0 |
| Better mean/median by absolute score | 0.1297 | 0.4366 | 1.7068 | 9 |
| Weighted mean, sample median if initial slope is zero | 0.1016 | 0.3315 | 1.3906 | 0 |

The last row is the selected start. It avoids a median calculation on ordinary
windows and remains within 4.2% of the original bursty timing at every size.
The unweighted median added 261/605/1203 bursty bisections. The weighted median
had none, but sorting the windows cost more. Selecting by absolute score can
prefer a saturated, nearly cancelling score far from the root.

The outlier count sums unfinished windows across 24 decompositions: length-512
Gaussian noise with a size-7 window, a sine with the default eight-stage
schedule, and a sine with the full schedule down to `1.5/L`; both uniform and
squared-triangular kernels; and positive spikes of 1e15, 1e30, 9.97e36 and
3.4e38 at index 128. Noise uses seed 42 and standard deviation 0.1; the sine
is `sin(2*pi*arange(512)/512)`. All use `SmoothAbs(0.2)`.
The selected solver finished every window in at most 42 iterations. Its full
squared-triangular schedule used 41 bisections at each outlier magnitude; the
uniform runs used none. The direct-score regression tests cover the same
extreme values, alongside lookup evaluation and worker parity.

For the seven-sample window `[0.1, -0.2, 0, big, 0.3, -0.1, 0.2]`, the selected
solver returns 0.10780207441813276 in three iterations with no bisections at
every outlier magnitude above. Research-score fixtures use 256 independent
bisections to resolve these roots. Constant `±finfo.max` windows return their
input, zero residual and zero iterations without probing outside the bracket.

With a median start alone, a finite Newton step near 0.37 was rejected by the
step-reduction safeguard, sending the estimate to the midpoint of a 3.4e38-wide
bracket. Over 100 halvings would be needed to recover a root near 0.5. Local
score probes now double their distance, up to `max_iter` probes per safeguard,
and tighten the bracket by score sign before bisection. The existing
step-reduction safeguard and positional stopping checks remain in place.

## GD parity and diagnostic cost

Both branches were compared with the pristine `dd3d739` checkout using 800
decompositions each. Components, residuals, iteration counts, final maximum
updates and finite-data convergence flags were exactly equal. Cases covered
lookup/direct scores, four kernels, three boundaries, lengths 1 through 512,
workers 1/2/3/4/8, explicit/default/full schedules, small and large scales,
constants, noise, spikes, bursts and separated clusters.

For length-4096 Gaussian noise with seed 42 and the default eight-stage
schedule, seven alternating timed runs after a warmup gave these medians:

| H | dd3d739, s | Issue #3, s | Issue #4, s |
| --- | ---: | ---: | ---: |
| 2 | 0.9815 | 0.9862 | 0.9937 |
| 0.5 | 2.4323 | 2.4202 | 2.4406 |

Issue #4 is 1.24% and 0.34% above the baseline, within run-to-run variation.
The GD score diagnostic needs no extra `psi` call. It reports the last
evaluated score, before the final update, and identical GD runs retain equal
stage diagnostics.
