# Accel performance suite

`accel-performance.yaml` covers workloads adapted from the 168 cases published at
https://docs.nvidia.com/cuml/latest/cuml-accel/benchmarks/.
It is separate from the broad `estimators.yaml` suite and requires no changes
to the dataset generators or harness.

## Provenance

The workload definitions were adapted from `cumlbench-dash` commit
`f5691fd26baf208e64ec4cdba6f3b080519e2adc`:

- `suites/cuml_accel_performance.json`: shapes, operations, inference training
  sizes, estimator parameters, and dataset settings. Row counts are rounded
  to two significant figures; feature and component counts are unchanged.
- `mlbench/backends/sklearn.py`: translation to estimator constructor arguments
  (including seeds, `n_init=1`, and `n_jobs=-1` where applicable).
- `policies/cuml_accel_performance_runtime.json`: complete-case CPU limits.

Explicit labels preserve the reference case names in runtime artifacts.
Content-derived identifiers remain independent of these display labels.

## Run

From the repository root, with the cuML development environment active:

```sh
python docs/benchmarks/run_cuml_accel_benchmarks.py \
  --output-dir /tmp/accel-performance
```

The driver requires exactly one CUDA-visible GPU, records actual system
metadata, runs both backends, and exports `benchmark-data.json`. Use
`--resume` with the same directory to retry incomplete/failed cases, or
`--export-only` to regenerate the publication JSON from existing artifacts
and their saved system metadata. See `docs/benchmarks/cuml-accel/README.md`
for the publication workflow. The suite can also be run directly using its
explicit path: `python/cuml/cuml/benchmark/suites/accel-performance.yaml`.

Both runs use one warmup and three measured repetitions. Compare median
measurement wall times, not warmups. Each case has a complete-case timeout
(70 seconds for small, 100 seconds for medium, and 850 seconds for large
workloads, rounded from the reference CPU limits). The manifest cannot express backend-specific
timeouts, so both backends use the reference CPU limits; the original GPU
limits were 180, 180, and 600 seconds respectively. The driver exports only
fully successful runs. Complete-case timeouts cannot establish per-operation
speedup bounds. If CPU cases time out, increase their limits and start a new
run directory rather than publishing deadline-based bounds.

The reference system was an NVIDIA RTX PRO 6000 Blackwell Workstation Edition
and AMD Ryzen Threadripper PRO 7975WX (32 physical cores), with about 134 GB
system memory. Packages: cuML 26.10.0a69, scikit-learn 1.9.0, hdbscan 0.8.44,
and umap-learn 0.5.12. Other systems and versions will produce different timings.

## Intentional differences

This suite reproduces workload coverage, not exact input arrays or results:

- **Blobs:** uses the existing scikit-learn generator with eight centers,
  rather than the reference's custom normal-distributed centers and samples.
- **Classification:** uses `make_classification` with the reference class and
  informative-feature counts, not the reference's noisy linear-signal labels.
- **Regression:** uses `make_regression` defaults, not the reference's
  16-informative-feature linear signal with noise 0.2. Signal, conditioning,
  and convergence can affect linear-model and forest timings.
- **Numeric:** uses the existing normal matrix generator. Exact random-number
  streams and array construction differ from the reference.
- **TargetEncoder:** uses the existing categorical generator (eight uniform
  categories and binary targets), not 64 Zipf-distributed categories and
  continuous targets. This is the least comparable dataset family.
- **Shapes:** measured and inference training row counts are rounded to two
  significant figures for readable workload definitions, rather than retaining
  exact byte-budget-derived sizes.
- **Training rows:** the reference generator internally split fitting cases
  90/10 and timed the training partition. This suite uses the full published
  row count (rounded) for fitting, consistent with this harness's shape
  semantics. Inference measurement and training row counts approximate the
  reference manifest.
- **Validation:** the harness verifies accel dispatch during warmup but does
  not implement the reference's operation-specific CPU/GPU parity checks.
  Timeout accounting therefore excludes those checks.

No suite-specific dataset-generation paths are required. Treat the results
as a comparable performance study, not a byte-for-byte or numerical replay.
