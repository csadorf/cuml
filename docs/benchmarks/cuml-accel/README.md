# Synchronizing the cuml.accel benchmark page

The checked-in `benchmark-data.json` is the compact input for the Sphinx page.
The current artifact contains the local `accel-performance` measurements from
2026-10-07 (cuML 26.12.0 and scikit-learn 1.9.1), replacing the earlier
`cumlbench-dash` reference. All 168 GPU cases and 166 CPU cases completed;
two CPU UMAP transforms timed out at their 1800-second complete-case limits.
Their CPU medians remain null, and rendering shows timeouts without speedup
bounds. Five CPU cases were retained from an interrupted first attempt and
the remainder completed using resume. Fresh
measurements can be produced locally using the accel performance suite and
the driver below; raw benchmark observations stay outside this repository. The file stores the benchmark system and package versions
alongside case labels, shapes, median timings, CPU timeout limits when
applicable, and PCA component counts that cannot be derived from labels.
Complete-case deadlines are not per-operation timing bounds and must not be
used to infer speedups. The strict driver below still requires every case to
succeed before exporting; the current partial-success artifact was assembled
from validated raw observations, with failures explicitly retained as null CPU
timings and complete-case limits. Numerical CPU/GPU parity was not checked.

Speedups, classifications, summaries, input sizes, and display units are
derived while rendering.

## Run and export fresh measurements

From the repository root, activate the development environment containing the
updated local cuML package and all suite dependencies (including scikit-learn,
umap-learn, and hdbscan). On Linux, select exactly one CUDA-visible GPU and run:

```console
CUDA_VISIBLE_DEVICES=0 python docs/benchmarks/run_cuml_accel_benchmarks.py \
  --output-dir /path/to/accel-performance-run
```

The Python driver invokes the specific `accel-performance.yaml` suite with
both CPU and `cuml.accel` backends, using one warmup and three measurements.
Do not enable `CUML_ACCEL_ENABLED` in the driver process; it launches isolated
backend processes itself. The output directory must initially be empty.
It contains `scikit-learn.json`, `cuml.accel.json`, a `system.json` snapshot,
and the exported `benchmark-data.json`. CPU topology and RAM are read from
Linux `/proc`, and GPU name and memory from CUDA. Package versions come from
the benchmark artifacts, not the reference publication.

Continue an interrupted run or regenerate only the compact publication file:

```console
python docs/benchmarks/run_cuml_accel_benchmarks.py \
  --output-dir /path/to/accel-performance-run --resume
python docs/benchmarks/run_cuml_accel_benchmarks.py \
  --output-dir /path/to/accel-performance-run --export-only
```

Resume requires the same system, software, and suite. Export-only uses saved
metadata and does not probe the current GPU. The exporter pairs observations
by workload ID and checks labels, execution settings, complete measurements,
and GPU dispatch evidence. Both input artifacts must satisfy the benchmark
artifact schema. All CPU and GPU cases must complete successfully; timeouts
and other failures prevent publication. Complete-case deadlines include
startup, preparation, fitting, warmups, and multiple measurements, so they
cannot establish per-operation speedup lower bounds. To complete a run with
CPU timeouts, increase the suite's timeouts and start a new output directory;
changed execution settings cannot resume existing artifacts. Neither running
nor exporting replaces the checked-in publication file automatically.

This is approximate reproduction: shapes and timeouts are rounded and inputs
use the existing benchmark generators. See
`python/cuml/cuml/benchmark/suites/accel-performance.md` for provenance and
methodological differences, including the absence of output-parity checks.

## Synchronize and render

From the repository root, synchronize an updated publication artifact with:

```console
python docs/benchmarks/generate_cuml_accel_benchmarks.py sync \
  --data /path/to/benchmark-data.json
```

The sync command accepts publication schema version 1. The artifact contains
168 cases: the selected 165-case performance grid plus three additional
medium-wide PCA component-rank measurements. The existing medium-wide PCA case
supplies the rank-1,024 point. The command validates and copies the artifact
unchanged, then renders the RST page and SVG heatmaps.

Render the page and heatmaps, or verify that they are current, with:

```console
python docs/benchmarks/generate_cuml_accel_benchmarks.py render
python docs/benchmarks/generate_cuml_accel_benchmarks.py sync --check \
  --data /path/to/benchmark-data.json
python docs/benchmarks/generate_cuml_accel_benchmarks.py render --check
```

`sync --check` validates both the supplied and checked-in artifacts, verifies
that they are byte-for-byte identical, and checks the rendered files without
writing.

Edit `docs/source/cuml-accel/benchmarks.rst.in` for narrative or structural
changes. Do not edit the generated `benchmarks.rst` or SVG files directly.
