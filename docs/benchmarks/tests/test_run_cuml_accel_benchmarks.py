# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test publication export without executing the performance workloads."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_cuml_accel_benchmarks.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_accel_benchmarks", SCRIPT)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


@pytest.fixture
def inputs():
    from cuml.benchmark.harness import _base_result, _run_record
    from cuml.benchmark.suite import load_suite

    plan = load_suite(driver.SUITE)
    system = json.loads(
        (SCRIPT.parent / "cuml-accel/benchmark-data.json").read_text()
    )["system"]
    artifacts = []
    for run in plan.runs:
        record = _run_record(run, ["test"])
        record["system"]["label"] = "test"
        packages = {p["name"]: p for p in record["software"]["packages"]}
        # Tests do not depend on the environment's installed optional packages.
        for package in packages.values():
            package["version"] = "1.0"
        results = []
        for case in run.cases:
            package = (
                "umap-learn"
                if case.estimator == "UMAP"
                else "hdbscan"
                if case.estimator == "HDBSCAN"
                else "scikit-learn"
            )
            result = _base_result(run, case, package, "1.0", packages[package])
            result["outcome"] = {"status": "success"}
            result["observations"] = [
                {
                    "role": "warmup",
                    "outcome": {"status": "success"},
                    "timings": [
                        {"name": "wall_time", "unit": "s", "value": 1000}
                    ],
                    "extensions": {
                        driver.ACCEL_EXTENSION: {
                            "gpu_calls": 1,
                            "cpu_calls": 0,
                        }
                    },
                },
                *[
                    {
                        "role": "measurement",
                        "outcome": {"status": "success"},
                        "timings": [
                            {"name": "wall_time", "unit": "s", "value": value}
                        ],
                        "extensions": {},
                    }
                    for value in (1, 3, 2)
                ],
            ]
            for sequence, observation in enumerate(result["observations"]):
                observation.update(
                    id=f"observation-{sequence}", sequence=sequence, metrics=[]
                )
            results.append(result)
        artifacts.append(
            {"schema_version": 2, "run": record, "results": results}
        )
    return (*artifacts, system, plan)


def test_export_complete_publication(inputs):
    cpu, gpu, system, plan = inputs
    gpu["results"].reverse()
    output = driver.export_data(cpu, gpu, system, plan)
    assert len(output["records"]) == 168
    assert all(
        r["cpu_median_sec"] == r["gpu_median_sec"] == 2
        for r in output["records"]
    )
    assert output["system"] == system
    assert set(output["packages"]) == driver.PACKAGES
    pca = next(
        r
        for r in output["records"]
        if r["case_label"] == "pca.fit_transform.medium.wide"
    )
    assert pca["components"] == 1024
    assert pca["rows"] == 61000


@pytest.mark.parametrize(
    "phase", ["preparation", "setup", "warmup", "timed_execution"]
)
def test_cpu_timeout_cannot_establish_operation_bound(inputs, phase):
    cpu, gpu, system, plan = inputs
    result = cpu["results"][0]
    result["outcome"] = {
        "status": "failed",
        "last_phase": phase,
        "error": {
            "type": "TimeoutExpired",
            "message": "Case exceeded 850 seconds",
        },
    }
    result["observations"] = []
    with pytest.raises(ValueError, match="cannot establish operation-time"):
        driver.export_data(cpu, gpu, system, plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "identity",
        "label",
        "cpu_failure",
        "gpu_failure",
        "partial",
        "duplicate_measurement",
        "nan",
        "fallback",
        "no_gpu_calls",
        "missing_warmup",
        "packages",
        "system",
        "plan",
        "timeout_limit",
    ],
)
def test_reject_invalid_artifacts(inputs, mutation):
    cpu, gpu, system, plan = inputs
    result = gpu["results"][0]
    if mutation == "missing":
        gpu["results"].pop()
    elif mutation == "duplicate":
        gpu["results"].append(copy.deepcopy(result))
    elif mutation == "identity":
        result["input"]["dimensions"][0]["size"] += 1
    elif mutation == "label":
        result["case_label"] = "wrong"
    elif mutation in {"cpu_failure", "gpu_failure"}:
        target = cpu["results"][0] if mutation == "cpu_failure" else result
        target["outcome"] = {
            "status": "failed",
            "error": {"type": "ValueError"},
        }
    elif mutation == "partial":
        result["observations"].pop()
    elif mutation == "duplicate_measurement":
        result["observations"][2] = copy.deepcopy(result["observations"][1])
    elif mutation == "nan":
        result["observations"][1]["timings"][0]["value"] = float("nan")
    elif mutation == "fallback":
        result["observations"][0]["extensions"][driver.ACCEL_EXTENSION][
            "cpu_calls"
        ] = 1
    elif mutation == "no_gpu_calls":
        result["observations"][0]["extensions"][driver.ACCEL_EXTENSION][
            "gpu_calls"
        ] = 0
    elif mutation == "missing_warmup":
        result["observations"].pop(0)
    elif mutation == "packages":
        gpu["run"]["software"]["packages"][1]["version"] = "different"
    elif mutation == "system":
        gpu["run"]["system"]["label"] = "different"
    elif mutation == "plan":
        gpu["run"]["extensions"][driver.EXTENSION]["execution_plan"][0][
            "repetitions"
        ] = 1
    else:
        cpu["results"][0]["outcome"] = {
            "status": "failed",
            "error": {"type": "TimeoutExpired"},
        }
        cpu["results"][0]["extensions"][driver.EXTENSION]["timeout_sec"] = 1
    with pytest.raises(ValueError):
        driver.export_data(cpu, gpu, system, plan)


def save_inputs(directory, inputs):
    cpu, gpu, system, _ = inputs
    directory.mkdir(exist_ok=True)
    for name, value in (
        ("scikit-learn.json", cpu),
        ("cuml.accel.json", gpu),
        (
            "system.json",
            {
                "publication_system": system,
                "benchmark_system": cpu["run"]["system"],
            },
        ),
    ):
        (directory / name).write_text(json.dumps(value))


def test_export_only_uses_saved_metadata(inputs, tmp_path, monkeypatch):
    save_inputs(tmp_path, inputs)
    monkeypatch.setattr(
        driver, "collect_system", lambda: pytest.fail("must not probe GPU")
    )
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must not run benchmarks"),
    )
    assert driver.main(["--output-dir", str(tmp_path), "--export-only"]) == 0
    assert (
        len(
            json.loads((tmp_path / "benchmark-data.json").read_text())[
                "records"
            ]
        )
        == 168
    )


def test_driver_invokes_both_backends(inputs, tmp_path, monkeypatch):
    from cuml.benchmark import harness

    output = tmp_path / "run"
    monkeypatch.setattr(harness, "_run_record", lambda *args: inputs[0]["run"])
    monkeypatch.setattr(driver, "collect_system", lambda: inputs[2])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        save_inputs(output, inputs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    # Fixture artifacts require _source subprocess calls only during construction.
    assert driver.main(["--output-dir", str(output)]) == 0
    assert calls[0].count("--implementation") == 2
    assert str(driver.SUITE) in calls[0]
    assert (output / "benchmark-data.json").exists()


def test_malformed_timeout_outcome_rejected(inputs):
    cpu, gpu, system, plan = inputs
    cpu["results"][0]["outcome"] = {
        "status": "failed",
        "error": {"type": "TimeoutExpired"},
    }
    with pytest.raises(ValueError, match="Invalid benchmark artifact"):
        driver.export_data(cpu, gpu, system, plan)


@pytest.mark.parametrize("change", ["software", "suite"])
def test_resume_rejects_stale_complete_artifacts(
    inputs, tmp_path, monkeypatch, change
):
    from cuml.benchmark import harness

    save_inputs(tmp_path, inputs)
    monkeypatch.setattr(driver, "collect_system", lambda: inputs[2])

    def current_record(run, argv):
        record = copy.deepcopy(
            inputs[0 if run.implementation == "scikit-learn" else 1]["run"]
        )
        if change == "software":
            record["software"]["packages"][0]["version"] = "new-version"
        else:
            record["extensions"][driver.EXTENSION]["execution_plan"][0][
                "case_label"
            ] = "changed"
        return record

    monkeypatch.setattr(harness, "_run_record", current_record)
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must reject before launch"),
    )
    assert driver.main(["--output-dir", str(tmp_path), "--resume"]) == 1
    assert not (tmp_path / "benchmark-data.json").exists()


def test_failed_benchmark_exit_does_not_export_stale_artifacts(
    inputs, tmp_path, monkeypatch
):
    from cuml.benchmark import harness

    output = tmp_path / "run"
    monkeypatch.setattr(harness, "_run_record", lambda *args: inputs[0]["run"])
    monkeypatch.setattr(driver, "collect_system", lambda: inputs[2])

    def run(*args, **kwargs):
        save_inputs(output, inputs)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.main(["--output-dir", str(output)]) == 1
    assert not (output / "benchmark-data.json").exists()


def test_export_rejects_unrelated_snapshot(inputs, tmp_path):
    save_inputs(tmp_path, inputs)
    snapshot = json.loads((tmp_path / "system.json").read_text())
    snapshot["benchmark_system"]["label"] = "unrelated machine"
    (tmp_path / "system.json").write_text(json.dumps(snapshot))
    assert driver.main(["--output-dir", str(tmp_path), "--export-only"]) == 1
    assert not (tmp_path / "benchmark-data.json").exists()


def test_resume_rejects_changed_hardware(inputs, tmp_path, monkeypatch):
    save_inputs(tmp_path, inputs)
    monkeypatch.setattr(driver, "collect_system", lambda: {"components": []})
    assert driver.main(["--output-dir", str(tmp_path), "--resume"]) == 1
    assert not (tmp_path / "benchmark-data.json").exists()


def test_collect_system_requires_one_visible_gpu(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=lambda: 2)
            )
        ),
    )
    with pytest.raises(ValueError, match="exactly one"):
        driver.collect_system()
