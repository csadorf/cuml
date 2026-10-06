cuml.benchmark
==============

Execution Harness
-----------------

.. automodule:: cuml.benchmark.harness
   :members: run_suite

Suite Configuration
-------------------

.. automodule:: cuml.benchmark.suite
   :members: Suite, ResolvedCase, SuiteError, resolve_case, load_suite, load_suite_reference, suite_profile_names, suite_manifest_json_schema

Execution Backends
------------------

.. automodule:: cuml.benchmark.backends.base
   :members:

Estimator Specifications
------------------------

.. automodule:: cuml.benchmark.registry
   :members: EstimatorSpec

Data Generation
---------------

.. automodule:: cuml.benchmark.datasets
   :members: generate_data, resolve_dataset, resolve_dtypes

Artifact Identity and Schema
----------------------------

.. automodule:: cuml.benchmark.identity
   :members: canonical_json, identity_preimage, result_id, CanonicalizationError

.. automodule:: cuml.benchmark.schemas
   :members: benchmark_result_schema

NVTX Profiling
--------------

.. automodule:: cuml.benchmark.nvtx_benchmark
   :members: Profiler
