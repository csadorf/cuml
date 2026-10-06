cuml.benchmark
==============

The built-in harness runs YAML-defined estimator workloads and writes JSON
observation artifacts. Start with the
:doc:`CLI benchmarking guide </developer_guide/benchmarking>` for commands,
suite/profile selection, manifest examples, timing semantics, and resume rules.

Suite loading and execution
---------------------------

For programmatic use, load a built-in suite name or a YAML path, then pass the
resolved plan's individual ``runs`` to the single-backend runner. The loader
accepts an optional ``implementations`` list; omission selects all declared
backends. Direct callers must isolate backend runs in separate processes and
configure backend startup. Prefer the CLI for automatic isolation, sequential
execution, and one output artifact per backend.

.. autofunction:: cuml.benchmark.suite.load_suite_reference

.. autofunction:: cuml.benchmark.harness.run_suite

Schemas
-------

Suite manifests require ``version: 2``. The generated JSON Schema describes the
strict manifest structure; suite loading additionally checks semantic constraints
such as inference inputs, backend compatibility, and duplicate workload IDs.

.. autofunction:: cuml.benchmark.suite.suite_manifest_json_schema

Result artifacts use ``schema_version: 2`` and contain ``run`` metadata and
``results`` with raw warmup/measurement observations. The canonical JSON Schema
is packaged at ``cuml/benchmark/schemas/benchmark-result.schema.json``; retrieve
the resource with:

.. autofunction:: cuml.benchmark.schemas.benchmark_result_schema
