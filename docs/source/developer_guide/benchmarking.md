# cuML Benchmark Suite

The suite-driven benchmark harness executes one implementation per process and
writes implementation-neutral, version-2 JSON artifacts. Workloads are defined
by strict version-2 YAML manifests. The legacy runners, version-1 manifests,
parameter-sweep CLI, and flat CSV exports are no longer supported.

## Running the benchmarks

Use an environment with this checkout's cuML Python package installed, PyYAML,
msgspec, NumPy, SciPy, and scikit-learn. Native cuML and cuml.accel additionally require
their GPU runtime dependencies; distributed cuML requires Dask dependencies and
at least two visible GPUs.

```bash
python -m cuml.benchmark --help
python -m cuml.benchmark --suite cuml_sg --profile smoke --output cuml.json
python -m cuml.benchmark --suite sklearn_cpu --profile smoke --output sklearn.json
python -m cuml.benchmark --suite cuml_accel --profile smoke --output accel.json
python -m cuml.benchmark --suite cuml_mg --profile smoke --output dask.json
```

The built-in suites are `cuml_sg` (the default), `sklearn_cpu`, `cuml_accel`,
and `cuml_mg`. Use `--suite /path/to/suite.yaml` for a custom manifest. Each
manifest selects its implementation; run implementations in separate processes.
The accel CLI activates acceleration at process startup before estimator imports.

| Option | Description |
|--------|-------------|
| `--suite` | Built-in suite name or version-2 YAML manifest path; defaults to `cuml_sg`. |
| `--profile` | Profile defined by the selected suite; defaults to `standard`. |
| `--output` | JSON artifact path; defaults to a timestamped file in the current directory. |
| `--resume` | Requires `--output`; retain successful cases from a compatible artifact and retry failures. |
| `-v`, `--verbose` | Include warmup and measurement repetition details in progress logs. |

The artifact is checkpointed atomically after each case. Case errors are recorded
and execution continues; the CLI exits nonzero if any case fails. Keyboard
interruption checkpoints the interrupted case before propagating. Resume requires
matching suite, software, system metadata, and workload identities.

Progress uses Python's `cuml.benchmark` logger: `INFO` for suite and case status,
`DEBUG` for individual repetitions, and `ERROR` for failures, including their
phase and reason. The CLI writes these logs to stderr; failure reasons are shown
without `--verbose`. Python callers can configure standard logging handlers and
levels instead of passing progress callbacks or verbosity to `run_suite`.
Subprocess logs are delivered through the parent, so the same handlers apply to
both execution modes. Handlers should be fast and nonblocking.

Logs are best-effort diagnostics, not a second results database. The artifact is
the authoritative record and is checkpointed before case completion is logged.
Worker phase tracking is independent of logging, so timeouts and crashes retain
the last phase received even when logs are disabled. A hard worker failure does
not recover observations that were not returned to the parent.

## Output and workload identity

Artifacts contain `schema_version: 2`, a `run` record, and a `results` list.
The run records the command, system components, software packages, and execution
plan. Each result describes an algorithm, dataset, operation, input, declared and
effective estimator parameters, implementation, outcome, and raw observations.
Warmups and measurements are recorded separately. Timing covers the estimator
operation and backend synchronization; data generation, estimator setup, and
artifact writes remain outside the timed interval. The harness does not call
`score()` or inspect outputs for heuristic quality metrics between repetitions.
Observation `metrics` lists are empty; correctness and quality checks belong in
separate tests. Accel warmup dispatch verification remains enabled.

The timing-only methodology is `cuml-benchmark-observations-v3`. The artifact
schema remains version 2 and workload identities are unchanged. Removing scoring
and output conversion changes the work performed between timed repetitions, so
resume rejects artifacts produced by the earlier `cuml-benchmark-observations-v2`
methodology rather than mixing measurements from the two protocols.

The packaged JSON Schema is available through
`cuml.benchmark.schemas.benchmark_result_schema()`. Workload IDs are computed by
`cuml.benchmark.identity.result_id()` from implementation-neutral fields. Derived
comparisons such as speedup belong in downstream analysis, not the artifact.
`ResolvedCase.id` is that same canonical workload ID, used for duplicate detection
and resume validation. `ResolvedCase` is the authoritative workload definition;
`ResolvedCase.to_artifact_fields()` serializes it into the neutral artifact schema,
excluding execution settings and runtime metadata. This serialization is also
used for canonical hashing, not maintained as a separate workload definition.
Short `case_label` values are derived from the canonical digest, not independently
hashed. Equivalent resolved workloads therefore have identical IDs and labels.
Descriptors and IDs are derived solely from resolved suite inputs, including
dataset defaults and profile scaling. Request resolution normalizes
generated/training/measured row counts, generator defaults, X/y dtypes, and
execution settings once. Data generation, partitioning, and serialization consume
those resolved settings rather than reinterpreting manifest shorthand. Inference identity also includes the
explicit inputs used by its untimed preparatory fit. Backend catalogs, estimator runtime
metadata, and measurements cannot change workload identity.

## Python requests and resolved cases

`resolve_case(request, profile)` accepts Python dictionaries with the same
structure as a YAML case and profile. Both entry points use the same strict
manifest schemas and semantic resolver; there is no separate Python request
model. Resolution does not import estimators or consult a backend. Suite loading
additionally validates backend catalog membership, backend profile requirements,
and duplicate workload identities.

```python
from cuml.benchmark.suite import resolve_case

request = {
    "estimator": "LogisticRegression",
    "operation": "predict",
    "input_selection": ["X"],
    "fit_input_selection": ["X", "y"],
    "parameters": {"max_iter": 100},
    "dataset": {
        "kind": "classification",
        "shape": {"rows": 100, "train_rows": 900, "features": 8},
        "dtype": "float64",
    },
}
profile = {"warmups": 1, "repetitions": 3, "size_scale": 1.0}
case = resolve_case(request, profile)
assert (case.generated_rows, case.training_rows, case.measured_rows) == (1000, 900, 100)

# Edit the request, then resolve again. The original case is unaffected.
request["dataset"]["shape"]["rows"] = 200
changed = resolve_case(request, profile)
```

Profile scaling, including the minimum of 32 scaled rows, applies identically to
Python and YAML requests. Omitted case timeouts inherit the profile value;
explicit `None` disables them. The resolver does not mutate requests and makes
independent copies of nested estimator parameters, dataset parameters, and dtypes.

`ResolvedCase` is a keyword-only, low-level container of explicit values. Its
constructor no longer accepts `rows`, `train_rows`, or `dtype`, and performs no
validation or normalization. It has no `__post_init__`; its lifecycle is a
read-only property derived from the operation. Normally construct it through
`resolve_case` or `load_suite`. Low-level callers must supply every resolved field
and maintain consistency themselves. `dataclasses.replace` does not re-resolve
dependent values; migrate request edits to the pattern above.

Field bindings are frozen, but nested mappings are not deeply immutable. Treat
resolved mappings as read-only. This separation does not change workload IDs,
artifact schema version 2, or timing methodology v3.

## YAML manifests

A complete case distinguishes dataset options from estimator options:

```yaml
version: 2
name: small-csr-example
implementation: scikit-learn
profiles:
  standard: {warmups: 1, repetitions: 1, size_scale: 1.0}
cases:
  - estimator: LogisticRegression
    dataset:
      kind: classification
      parameters: {n_classes: 2, n_informative: 5, density: 0.25}
      shape: {rows: 128, features: 8, train_rows: 128}
      dtype: float64
      format: csr
    operation: predict
    input_selection: [X]
    fit_input_selection: [X, y]
    parameters: {max_iter: 100}
```

All five top-level fields are required. Registered implementations include
`scikit-learn`, `cuml`, `cuml.accel`, and `cuml.dask`. All backends share a fixed,
closed manifest schema: backend registration cannot add profile fields. msgspec
checks required and unknown fields, strict types, defaults, and numeric bounds
before semantic resolution. Counts must be integers, not booleans; scalar scales
and timeouts also reject booleans and nonfinite values. Every profile is checked
structurally, including unselected profiles.

`cuml.benchmark.suite.suite_manifest_json_schema()` returns this common version-2
manifest JSON Schema as a dictionary, with no backend argument. It describes
structure, not registry membership, supported operations, generator constraints,
or duplicate workload identities. The loader checks those semantics separately,
and calls the selected backend's `validate_profile(profile_name, warmups)` hook
only for the selected profile. Structural errors include the suite path and the
invalid field's path; semantic errors identify the profile or case.

Suite imports and CLI help remain lightweight. Loading, profile discovery, and
manifest schema generation require msgspec; YAML loading also requires PyYAML.

- Each case requires `estimator`, `dataset`, `operation`, `input_selection`,
  and `parameters`.
  Operations are `fit`, `fit_predict`, `fit_transform`, `predict`, `transform`,
  `kneighbors`, and `score_samples`; estimators must implement the requested method.
- Required `input_selection` declares the measured arguments as `[X]`, `[y]`,
  or `[X, y]`. Inference cases additionally require `fit_input_selection` with
  the same allowed values to declare untimed preparatory fitting arguments.
  Fitting cases must omit `fit_input_selection` (or set it to `null`). There are
  no estimator-specific input defaults. LabelEncoder/LabelBinarizer declare
  `[y]`; supervised fitting typically declares `[X, y]` and prediction `[X]`.
  Execution consumes exactly the inputs declared in the suite.
- `dataset` requires `kind` and `shape`. Shape requires positive integer `rows`
  and `features`. Inference additionally requires positive integer `train_rows`;
  training operations must omit it.
- `dataset.parameters` is optional and separate from case `parameters`, which
  is passed to the estimator. Generator defaults resolve after scaling.
- `dataset.dtype` defaults to `float32` for both X and y. A mapping must specify
  both, for example `dtype: {X: int32, y: int64}`. Supported element types are
  `float32`, `float64`, `int32`, and `int64`.
- `dataset.format` defaults to `dense`; `csr` generates SciPy CSR features before
  backend conversion and contiguous training/inference slicing.
- `rows` always specifies the input size of the measured operation. Inference
  generates `train_rows + rows` samples together, then takes disjoint contiguous
  training and inference partitions. There is no implicit training/test split.
  To migrate an old inference case without `train_rows`, replace its total
  `rows: N` with `train_rows: round(0.9 * N)` and
  `rows: N - round(0.9 * N)`. Profiles scale the two sizes independently, so
  small smoke workloads can change because measured rows have a minimum of 32.
- Training operations (`fit`, `fit_predict`, `fit_transform`) use a fresh
  estimator for every warmup and measurement. Inference constructs and fits one
  estimator before the first observation, then reuses it for all warmups and
  measurements, including when there are no warmups. Construction and
  preparatory fitting are untimed; only the requested operation is measured.
- Profiles require nonnegative `warmups`, positive `repetitions`, and
  `size_scale` in `(0, 1]`. Scaling affects rows (minimum 32) and explicit training
  rows (minimum 1), not features.
- Optional `timeout_sec` can be set per profile or overridden per case. Numeric
  values must be finite and positive. An omitted case timeout inherits the
  profile value; explicit `null` disables the timeout. Timed cases run in
  subprocesses; Dask case timeouts are unsupported.
- Accel requires at least one warmup and automatically verifies dispatch on
  every warmup repetition. Evidence is recorded in `com.nvidia.cuml.accel` with
  `scope: warmup` and a zero-based `repetition`. CPU fallback fails the case.
  Measurement repetitions remain unprofiled; verification is not a suite option.
  Verification covers the requested operation, not inference's preparatory fit.

Supported generator options (generation uses seed 42):

| Dataset kind | Defaults and constraints |
|--------------|--------------------------|
| `blobs` | `centers: 5` (positive integer); `cluster_std: 1.0` (positive finite scalar) |
| `classification` | `n_classes: 2`, `n_informative: max(2, min(features, features // 2 + 1))`, `n_redundant: 0`, `n_clusters_per_class: 2`; informative + redundant must fit features, and classes x clusters-per-class must be at most `2 ** n_informative` |
| `matrix`, `regression`, `positive`, `categorical` | No dataset-specific options |
| CSR `classification` or `matrix` only | Additionally `density: 0.1`, a finite scalar in `(0, 1]`; forbidden for dense inputs |

CSR density is a target Bernoulli retention probability, not an exact realized
fraction. Masking uses a separate seed-42 RNG and leaves labels unchanged.
Generation allocates dense features and a dense mask before constructing CSR;
this is not a memory-scalable sparse generator. Distributed cuML CSR is rejected
before cluster creation. Other estimator/CSR combinations may fail during
execution even when the dataset request is valid.

Input types come from the dataset request, not the estimator name. CategoricalNB
suite cases explicitly request `dtype: {X: int32, y: int64}`. LabelEncoder and
LabelBinarizer consume labels rather than features. Workload identity includes
resolved generator settings, input format, and the dtype consumed by the
estimator. Actual feature or label dtype and resolved generator parameters are
also recorded in the `com.nvidia.cuml.benchmark` result extension.

## Capability examples

These custom-path suites are capability checks, not performance baselines or
built-in aliases:

```bash
python -m cuml.benchmark \
  --suite python/cuml/cuml/benchmark/suites/capabilities_sklearn.yaml \
  --profile standard --output /tmp/capabilities-sklearn.json
python -m cuml.benchmark \
  --suite python/cuml/cuml/benchmark/suites/capabilities_cuml.yaml \
  --profile standard --output /tmp/capabilities-cuml.json
python -m cuml.benchmark \
  --suite python/cuml/cuml/benchmark/suites/capabilities_accel.yaml \
  --profile standard --output /tmp/capabilities-accel.json
```

All three contain the same eight workloads: KMeans fit with custom blobs,
multiclass LogisticRegression fit, PCA fit with float32/float64 matrices, and
LogisticRegression fit/predict with float32/float64 CSR inputs. The accel example
verifies dispatch during every warmup. Equivalent workloads have matching
canonical IDs despite different implementation metadata and timings. This does
not advertise CSR or multiclass support for every estimator.

Representative dense-fit and CSR-inference smoke tests check actual input types
and values, schema validity, workload IDs, and accel dispatch across sklearn,
cuML, and cuml.accel. All capability manifests are also validated without running
every workload:

```bash
python -m pytest -q python/cuml/tests/test_benchmark.py
```

## Adding algorithm coverage

Add estimator specifications to `python/cuml/cuml/benchmark/registry.py` and
cases to the appropriate `benchmark/suites/` manifests. Include coverage for
each applicable implementation. For a new C++ algorithm, add a Google Benchmark
case under `cpp/bench/sg` and list its source in `cpp/bench/CMakeLists.txt`.

## Adding an execution backend

Registration is explicit, not plugin discovery. The interface is defined in
`python/cuml/cuml/benchmark/backends/base.py`.

1. Define a catalog mapping estimator names to `EstimatorSpec(module, name,
   package)` in `benchmark/registry.py`. Declare custom estimator inputs with
   `input_selection` and, for inference, `fit_input_selection` in the suite.
   Backend metadata never selects execution arguments.
2. Implement a `Backend` subclass in `benchmark/backends/`. Keep estimator,
   CUDA, distributed, and accelerator imports inside hooks, not at module scope.
   `GPUBackend` adds CUDA synchronization.
3. Add an instance to `BACKENDS` in `benchmark/backends/__init__.py`. Do not add
   implementation branches to the CLI, suite parser, or harness.
4. Add a version-2 manifest and focused registration, lifecycle, and artifact
   tests, following the `registered_suite` fixture and CLI integration tests in
   `python/cuml/tests/test_benchmark.py`.

Override only the hooks needed by the backend:

- `validate_profile(profile_name, warmups) -> None` checks backend semantics
  during loading, after common structural validation. Raise `SuiteError` for an
  unsupported selected profile; do not extend the manifest schema or return
  backend options.
- `bootstrap_process` and `prepare_process` establish process prerequisites
  before the CLI imports the harness. Preserve accelerator startup ordering.
- `runtime` owns resources and yields the value passed to `construct_estimator`.
- `load_estimator` resolves classes lazily; `construct_estimator` must not mutate
  declared parameters. `convert_data` receives the complete generated dataset
  before training/inference partitioning.
- `synchronize` completes work at timing boundaries. `instrumentation` surrounds
  the operation and yields a `VerificationResult` containing optional evidence
  and failure information. Accel profiles every warmup, never measurements.
- `software_packages` supplies sorted package records, used consistently in run
  and result metadata and resume compatibility checks.

The harness owns data generation, estimator lifecycle, timing, process isolation,
failure reporting, checkpointing, and neutral-v2 serialization. Backend evidence
belongs in namespaced extensions. Registration must also exist in spawned workers;
a temporary in-process mapping change is insufficient for timeout execution.

## NVTX profiling

`cuml.benchmark.nvtx_benchmark` remains available as a standalone Nsight Systems
profiling utility. It requires `nsys` version 2021.4 or later and profiles NVTX
ranges in the command supplied as its single argument:

```bash
python -m cuml.benchmark.nvtx_benchmark \
  "python -m cuml.benchmark --suite cuml_sg --profile smoke --output cuml.json"
```

It is independent of the harness's observation timing and JSON serialization.
