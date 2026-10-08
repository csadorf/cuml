cuml.benchmark
==============

The built-in harness runs YAML-defined estimator workloads and writes JSON
results files. Start with the
:doc:`CLI benchmarking guide </developer_guide/benchmarking>` for commands,
suite/profile selection, manifest examples, timing semantics, and resume rules.

Programmatic execution
----------------------

The public runner works directly in scripts and notebook cells, using the same
isolated provider execution as the CLI.

.. code-block:: python

    from cuml import benchmark

    artifacts = benchmark.run(
        "my-suite.yaml", providers=["cuml", "scikit-learn"]
    )

Results are JSON dictionaries keyed by provider. The defaults are the
``estimators`` suite, ``standard`` profile, and ``cuml`` provider. Customize
workloads and execution counts through YAML.

Set ``output="comparison"`` to save results and checkpoints; use ``resume=True``
to continue an existing run. Omitting ``output`` uses temporary storage.

.. autofunction:: cuml.benchmark.run

.. autoclass:: cuml.benchmark.BenchmarkRunError

Low-level suite loading and execution
-------------------------------------

The loader's ``providers=None`` selects all declared providers, unlike the
public runner and CLI. Direct callers of the single-provider harness must
isolate processes and configure backend startup themselves.

.. autofunction:: cuml.benchmark.suite.load_suite_reference

.. autofunction:: cuml.benchmark.harness.run_suite

Schemas
-------

The generated JSON Schema describes the strict suite manifest structure.
Suite loading additionally checks semantic constraints
such as inference inputs, provider compatibility, and duplicate workload IDs.

.. autofunction:: cuml.benchmark.suite.suite_manifest_json_schema

Results files contain ``run`` metadata and ``results`` with raw
warmup/measurement observations. The canonical JSON Schema
is packaged at ``cuml/benchmark/schemas/benchmark-result.schema.json``; retrieve
the resource with:

.. autofunction:: cuml.benchmark.schemas.benchmark_result_schema
