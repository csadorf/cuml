# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cuml.benchmark import harness
from cuml.benchmark.registry import EstimatorSpec
from cuml.benchmark.suite import ResolvedCase, Suite


class Estimator:
    calls = 0

    def __init__(self, **parameters):
        self.parameters = parameters

    def get_params(self, deep=False):
        return {**self.parameters, "effective_default": 1}

    def fit(self, X):
        type(self).calls += 1
        return self


def _suite(*, verify=False, warmups=1, repetitions=3, cases=None):
    case = ResolvedCase(
        "PCA",
        "matrix",
        "fit",
        64,
        8,
        {"n_components": 2},
        warmups,
        repetitions,
    )
    return Suite(
        Path("suite.yaml"),
        "test",
        "cuml.accel",
        "smoke" if verify else "standard",
        verify,
        tuple(cases or [case]),
    )


def _patch_case(monkeypatch):
    Estimator.calls = 0
    monkeypatch.setattr(
        harness,
        "_import_estimator",
        lambda suite, case: (
            Estimator,
            EstimatorSpec("sklearn.decomposition", "PCA", "scikit-learn"),
        ),
    )
    monkeypatch.setattr(
        harness,
        "_generate_data",
        lambda case, implementation: (
            np.ones((case.rows, case.features)),
            np.zeros(case.rows),
        ),
    )
    syncs = []
    monkeypatch.setattr(
        harness,
        "_synchronize",
        lambda implementation, value=None: syncs.append(value),
    )
    monkeypatch.setattr(harness, "_version", lambda package: "1.2.3")
    monkeypatch.setattr(harness, "_source", lambda: None)
    return syncs


def test_standard_accel_never_imports_or_enters_profiler(monkeypatch):
    syncs = _patch_case(monkeypatch)
    original = harness.importlib.import_module

    def guarded(name):
        assert name != "cuml.accel"
        return original(name)

    monkeypatch.setattr(harness.importlib, "import_module", guarded)
    result = harness.run_case(_suite(), _suite().cases[0])
    assert result["outcome"] == {"status": "success"}
    assert [o["role"] for o in result["observations"]] == [
        "warmup",
        "measurement",
        "measurement",
        "measurement",
    ]
    assert all(
        harness.ACCEL_EXTENSION not in o["extensions"]
        for o in result["observations"]
    )
    assert Estimator.calls == 4
    assert len(syncs) == 8
    assert result["parameters"]["effective"]["effective_default"] == 1


class Profile:
    def __init__(self, cpu_calls=0):
        self.method_calls = {
            "PCA.fit": SimpleNamespace(
                gpu_calls=1,
                gpu_time=0.01,
                cpu_calls=cpu_calls,
                cpu_time=0.02 if cpu_calls else 0,
                fallback_reasons={"unsupported"} if cpu_calls else set(),
            )
        }


class Profiler:
    def __init__(self, cpu_calls=0):
        self.cpu_calls = cpu_calls
        self.enters = 0

    @contextlib.contextmanager
    def profile(self, quiet=False):
        assert quiet
        self.enters += 1
        yield Profile(self.cpu_calls)


def test_smoke_profiles_only_first_warmup_and_gpu_dispatch_measures(
    monkeypatch,
):
    _patch_case(monkeypatch)
    profiler = Profiler()
    original = harness.importlib.import_module
    monkeypatch.setattr(
        harness.importlib,
        "import_module",
        lambda name: profiler if name == "cuml.accel" else original(name),
    )
    suite = _suite(verify=True, warmups=2, repetitions=2)
    result = harness.run_case(suite, suite.cases[0])
    assert result["outcome"]["status"] == "success"
    assert profiler.enters == 1
    assert [
        harness.ACCEL_EXTENSION in o["extensions"]
        for o in result["observations"]
    ] == [True, False, False, False]
    assert (
        len([o for o in result["observations"] if o["role"] == "measurement"])
        == 2
    )


def test_cpu_fallback_fails_and_omits_measurements(monkeypatch):
    _patch_case(monkeypatch)
    profiler = Profiler(cpu_calls=1)
    original = harness.importlib.import_module
    monkeypatch.setattr(
        harness.importlib,
        "import_module",
        lambda name: profiler if name == "cuml.accel" else original(name),
    )
    suite = _suite(verify=True)
    result = harness.run_case(suite, suite.cases[0])
    assert result["outcome"]["error"]["type"] == "CpuFallback"
    assert [o["role"] for o in result["observations"]] == ["warmup"]
    assert (
        result["observations"][0]["extensions"][harness.ACCEL_EXTENSION][
            "cpu_calls"
        ]
        == 1
    )
    assert Estimator.calls == 1


def test_case_exception_is_schema_shaped_failure(monkeypatch):
    _patch_case(monkeypatch)
    monkeypatch.setattr(
        Estimator,
        "fit",
        lambda self, X: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    suite = _suite()
    result = harness.run_case(suite, suite.cases[0])
    assert result["outcome"]["status"] == "failed"
    assert result["outcome"]["error"] == {
        "type": "RuntimeError",
        "message": "boom",
    }
    assert result["observations"][0]["outcome"]["status"] == "failed"


def test_atomic_checkpoint_after_every_case_and_continue_failures(monkeypatch):
    cases = [
        ResolvedCase("PCA", "matrix", "fit", 32, 4, {}, 1, 1),
        ResolvedCase("PCA", "matrix", "fit", 64, 4, {}, 1, 1),
    ]
    suite = _suite(cases=cases)
    monkeypatch.setattr(
        harness, "_run_record", lambda suite, argv: {"created_at": "now"}
    )
    monkeypatch.setattr(
        harness, "_runtime", lambda suite: contextlib.nullcontext()
    )
    statuses = iter(("failed", "success"))
    monkeypatch.setattr(
        harness,
        "run_case",
        lambda suite, case, client=None: {
            "outcome": {"status": next(statuses)}
        },
    )
    writes = []
    monkeypatch.setattr(
        harness,
        "atomic_write",
        lambda path, artifact: writes.append(
            (len(artifact["results"]), "completed_at" in artifact["run"])
        ),
    )
    artifact = harness.run_suite(suite, "artifact.json", ["benchmark"])
    assert [r["outcome"]["status"] for r in artifact["results"]] == [
        "failed",
        "success",
    ]
    assert writes == [(0, False), (1, False), (2, False), (2, True)]
