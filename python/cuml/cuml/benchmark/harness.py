# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution and neutral-v2 artifact serialization."""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .registry import estimator_spec
from .suite import ResolvedCase, Suite, SuiteError

METHODOLOGY = "cuml-benchmark-observations-v2"
EXTENSION = "com.nvidia.cuml.benchmark"
ACCEL_EXTENSION = "com.nvidia.cuml.accel"
ProgressCallback = Callable[[str], None]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return repr(value)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _source() -> dict[str, Any]:
    for root in (Path.cwd(), Path(__file__).resolve().parent):
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
            ).stdout.strip()
            repository = (
                subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5,
                ).stdout.strip()
                or None
            )
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                ).stdout
            )
            return {
                "repository": repository,
                "revision": revision,
                "dirty": dirty,
            }
        except Exception:
            pass
    # Wheels retain the exact source commit even when no checkout is available.
    commit_file = Path(__file__).resolve().parents[1] / "GIT_COMMIT"
    try:
        revision = commit_file.read_text(encoding="utf-8").strip()
    except OSError:
        revision = None
    return {
        "repository": "https://github.com/rapidsai/cuml.git",
        "revision": revision,
        "dirty": None,
    }


def _gpu_components() -> list[dict[str, Any]]:
    try:
        cp = importlib.import_module("cupy")
        count = cp.cuda.runtime.getDeviceCount()
        devices: dict[str, int] = {}
        for index in range(count):
            props = cp.cuda.runtime.getDeviceProperties(index)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            devices[str(name)] = devices.get(str(name), 0) + 1
        return [
            {
                "type": "accelerator",
                "name": name,
                "count": count,
                "attributes": {"vendor": "NVIDIA"},
            }
            for name, count in devices.items()
        ]
    except Exception:
        return []


def _run_record(suite: Suite, argv: list[str]) -> dict[str, Any]:
    package_names = {
        estimator_spec(suite.implementation, c.estimator).package
        for c in suite.cases
    }
    if suite.implementation in {"cuml", "cuml.dask", "cuml.accel"}:
        package_names.add("cuml")
    cpu = platform.processor() or platform.machine() or "unknown processor"
    return {
        "id": f"urn:uuid:{uuid.uuid4()}",
        "created_at": _now(),
        "methodology": {"id": METHODOLOGY},
        "command": {"argv": argv, "cwd": os.getcwd()},
        "system": {
            "label": socket.gethostname(),
            "components": [
                {
                    "type": "processor",
                    "name": cpu,
                    "count": os.cpu_count() or 1,
                    "attributes": {"platform": platform.platform()},
                },
                *_gpu_components(),
            ],
        },
        "software": {
            "runtimes": [
                {"name": "python", "version": platform.python_version()}
            ],
            "packages": [
                {
                    "name": p,
                    "version": _version(p),
                    "build": None,
                    "source": _source() if p == "cuml" else None,
                }
                for p in sorted(package_names)
            ],
        },
        "extensions": {
            EXTENSION: {
                "suite": suite.name,
                "suite_path": str(suite.path),
                "profile": suite.profile_name,
                "implementation": suite.implementation,
                "verify_accel_dispatch": suite.verify_accel_dispatch,
            }
        },
    }


def atomic_write(path: str | Path, artifact: dict[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                artifact, stream, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _generate_data(case: ResolvedCase, implementation: str):
    # Importing sklearn is deliberately delayed until the CLI has bootstrapped
    # cuml.accel (or verified isolation for a non-accel suite).
    datasets = importlib.import_module("sklearn.datasets")
    seed = 42
    if case.dataset == "classification":
        informative = max(2, min(case.features, case.features // 2 + 1))
        X, y = datasets.make_classification(
            n_samples=case.rows,
            n_features=case.features,
            n_informative=informative,
            n_redundant=0,
            random_state=seed,
        )
    elif case.dataset == "regression":
        X, y = datasets.make_regression(
            n_samples=case.rows,
            n_features=case.features,
            random_state=seed,
        )
    elif case.dataset == "blobs":
        X, y = datasets.make_blobs(
            n_samples=case.rows,
            n_features=case.features,
            centers=5,
            random_state=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(case.rows, case.features))
        y = rng.integers(0, 3, size=case.rows)
        if case.dataset == "positive":
            X = np.abs(X)
        elif case.dataset == "categorical":
            X = rng.integers(0, 8, size=(case.rows, case.features))
            y = rng.integers(0, 2, size=case.rows)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if case.estimator in {"MultinomialNB", "ComplementNB", "CategoricalNB"}:
        X = (
            np.abs(X)
            if case.estimator != "CategoricalNB"
            else np.asarray(X, dtype=np.int32)
        )
    if case.dataset == "categorical" and case.estimator not in {
        "LabelEncoder",
        "LabelBinarizer",
    }:
        pandas = importlib.import_module("pandas")
        X = pandas.DataFrame(
            X, columns=[f"feature_{index}" for index in range(case.features)]
        )
    if implementation == "cuml.dask":
        da = importlib.import_module("dask.array")
        chunks = (max(1, case.rows // 2), case.features)
        if case.dataset == "categorical" and case.estimator not in {
            "LabelEncoder",
            "LabelBinarizer",
        }:
            cudf = importlib.import_module("cudf")
            dask_cudf = importlib.import_module("dask_cudf")
            X = dask_cudf.from_cudf(cudf.from_pandas(X), npartitions=2)
        else:
            X = da.from_array(X, chunks=chunks)
        y = da.from_array(y, chunks=(chunks[0],))
    return X, y


def _import_estimator(suite: Suite, case: ResolvedCase):
    spec = estimator_spec(suite.implementation, case.estimator)
    module = importlib.import_module(spec.module)
    return getattr(module, spec.name), spec


def _inputs(case: ResolvedCase, spec, X, y):
    if case.estimator in {"LabelEncoder", "LabelBinarizer"}:
        return (y,)
    return (X, y) if spec.supervised else (X,)


def _invoke(estimator, case: ResolvedCase, args):
    operation = case.operation
    if operation in {"fit", "fit_predict", "fit_transform"}:
        return getattr(estimator, operation)(*args)
    estimator.fit(*args)
    X = args[0]
    if operation == "kneighbors":
        return estimator.kneighbors(X)
    return getattr(estimator, operation)(X)


def _synchronize(implementation: str, value: Any = None) -> None:
    if implementation == "cuml.dask" and value is not None:
        try:
            importlib.import_module("dask.distributed").wait(value)
        except (TypeError, AttributeError):
            if hasattr(value, "compute"):
                value.compute()
    if implementation in {"cuml", "cuml.dask", "cuml.accel"}:
        try:
            importlib.import_module("cupy").cuda.runtime.deviceSynchronize()
        except Exception:
            # Preserve the estimator's original exception path on machines where
            # synchronization is unavailable; execution itself will fail clearly.
            pass


def _metric(
    estimator, case: ResolvedCase, X, y, output
) -> list[dict[str, Any]]:
    try:
        if (
            hasattr(estimator, "score")
            and estimator_spec("scikit-learn", case.estimator).supervised
        ):
            value = float(estimator.score(X, y))
            return [
                {
                    "name": "score",
                    "value": value,
                    "unit": "ratio",
                    "direction": "higher_is_better",
                }
            ]
    except Exception:
        pass
    try:
        candidate = output[0] if isinstance(output, tuple) else output
        if hasattr(candidate, "compute"):
            candidate = candidate.compute()
        if hasattr(candidate, "get"):
            candidate = candidate.get()
        array = np.asarray(candidate)
        value = float(np.isfinite(array).mean()) if array.size else 1.0
        return [
            {
                "name": "finite_fraction",
                "value": value,
                "unit": "ratio",
                "direction": "higher_is_better",
            }
        ]
    except Exception:
        return []


def _dispatch_evidence(profile_result) -> dict[str, Any]:
    calls = []
    for name, stats in sorted(profile_result.method_calls.items()):
        calls.append(
            {
                "function": name,
                "gpu_calls": stats.gpu_calls,
                "gpu_time_s": stats.gpu_time,
                "cpu_calls": stats.cpu_calls,
                "cpu_time_s": stats.cpu_time,
                "fallback_reasons": sorted(stats.fallback_reasons),
            }
        )
    return {
        "scope": "first_warmup",
        "calls": calls,
        "gpu_calls": sum(item["gpu_calls"] for item in calls),
        "cpu_calls": sum(item["cpu_calls"] for item in calls),
    }


def _failure(exc: BaseException, phase: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "last_phase": phase,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _base_result(
    suite: Suite, case: ResolvedCase, package: str, version: str
) -> dict[str, Any]:
    return {
        "id": case.id,
        "algorithm": case.estimator,
        "dataset": case.dataset,
        "operation": case.operation,
        "input": {
            "dimensions": [
                {"name": "rows", "size": case.rows},
                {"name": "features", "size": case.features},
            ],
            "data_type": "float32",
            "attributes": {},
        },
        "parameters": {
            "declared": _jsonable(case.parameters),
            "effective": {},
        },
        "implementation": {
            "name": package,
            "version": version,
            "build": None,
            "source": _source() if package == "cuml" else None,
        },
        "outcome": {"status": "failed", "last_phase": "preparation"},
        "observations": [],
        "extensions": {
            EXTENSION: {
                "implementation": suite.implementation,
                "profile": suite.profile_name,
            }
        },
    }


def run_case(
    suite: Suite,
    case: ResolvedCase,
    client: Any = None,
    *,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    spec = estimator_spec(suite.implementation, case.estimator)
    result = _base_result(suite, case, spec.package, _version(spec.package))
    phase = "preparation"
    try:
        estimator_class, spec = _import_estimator(suite, case)
        X, y = _generate_data(case, suite.implementation)
        args = _inputs(case, spec, X, y)
        observations = result["observations"]
        sequence = 0
        for role, count in (
            ("warmup", case.warmups),
            ("measurement", case.repetitions),
        ):
            for repetition in range(count):
                phase = "setup"
                constructor_parameters = dict(case.parameters)
                if suite.implementation == "cuml.dask":
                    constructor_parameters["client"] = client
                estimator = estimator_class(**constructor_parameters)
                if not result["parameters"]["effective"] and hasattr(
                    estimator, "get_params"
                ):
                    result["parameters"]["effective"] = _jsonable(
                        estimator.get_params(deep=False)
                    )
                phase = role if role == "warmup" else "timed_execution"
                profile_result = None
                manager = contextlib.nullcontext()
                if (
                    suite.verify_accel_dispatch
                    and role == "warmup"
                    and repetition == 0
                ):
                    # This is the only profiler import in the harness. Standard
                    # accel runs cannot initialize or enter it.
                    manager = importlib.import_module("cuml.accel").profile(
                        quiet=True
                    )
                with manager as profile_result:
                    _synchronize(suite.implementation)
                    start = time.perf_counter()
                    output = _invoke(estimator, case, args)
                    _synchronize(suite.implementation, output)
                    elapsed = time.perf_counter() - start
                observation = {
                    "id": f"{role}-{repetition}",
                    "sequence": sequence,
                    "role": role,
                    "outcome": {"status": "success"},
                    "timings": [
                        {"name": "wall_time", "value": elapsed, "unit": "s"}
                    ],
                    "extensions": {},
                }
                sequence += 1
                if role == "measurement":
                    metrics = _metric(estimator, case, X, y, output)
                    if metrics:
                        observation["metrics"] = metrics
                if profile_result is not None:
                    evidence = _dispatch_evidence(profile_result)
                    observation["extensions"][ACCEL_EXTENSION] = evidence
                    if evidence["cpu_calls"]:
                        error = RuntimeError(
                            "cuml.accel dispatch verification recorded CPU fallback"
                        )
                        observation["outcome"] = {
                            "status": "failed",
                            "last_phase": "dispatch_verification",
                            "error": {
                                "type": "CpuFallback",
                                "message": str(error),
                            },
                        }
                        observations.append(observation)
                        if verbose and progress is not None:
                            progress(
                                f"  {role} {repetition + 1}/{count} failed "
                                f"in {elapsed:.3f}s"
                            )
                        result["outcome"] = {
                            "status": "failed",
                            "last_phase": "dispatch_verification",
                            "error": {
                                "type": "CpuFallback",
                                "message": str(error),
                            },
                        }
                        return result
                observations.append(observation)
                if verbose and progress is not None:
                    progress(
                        f"  {role} {repetition + 1}/{count} completed "
                        f"in {elapsed:.3f}s"
                    )
        result["outcome"] = {"status": "success"}
        return result
    except Exception as exc:
        error = _failure(exc, phase)
        role = "warmup" if phase == "warmup" else "measurement"
        result["observations"].append(
            {
                "id": f"{role}-failed",
                "sequence": len(result["observations"]),
                "role": role,
                "outcome": error,
                "extensions": {
                    EXTENSION: {"traceback": traceback.format_exc()}
                },
            }
        )
        if verbose and progress is not None:
            progress(f"  {role} failed: {type(exc).__name__}: {exc}")
        result["outcome"] = error
        return result


@contextlib.contextmanager
def _runtime(suite: Suite):
    if suite.implementation != "cuml.dask":
        yield None
        return
    cp = importlib.import_module("cupy")
    count = cp.cuda.runtime.getDeviceCount()
    if count < 2:
        raise SuiteError(
            f"cuml.dask suite requires at least two visible GPUs; found {count}"
        )
    LocalCUDACluster = importlib.import_module("dask_cuda").LocalCUDACluster
    Client = importlib.import_module("dask.distributed").Client
    with LocalCUDACluster(n_workers=count, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            yield client


def run_suite(
    suite: Suite,
    output: str | Path,
    argv: list[str] | None = None,
    *,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    artifact = {
        "schema_version": 2,
        "run": _run_record(suite, list(argv or sys.argv)),
        "results": [],
    }
    atomic_write(output, artifact)
    suite_started = time.perf_counter()
    total = len(suite.cases)
    if progress is not None:
        progress(
            f"Running suite {suite.name!r}, profile {suite.profile_name!r} "
            f"({total} cases)"
        )
    with _runtime(suite) as client:
        for index, case in enumerate(suite.cases, start=1):
            label = (
                f"{case.estimator}.{case.operation} "
                f"({case.rows}x{case.features})"
            )
            if progress is not None:
                progress(f"[{index}/{total}] {label}: starting")
            case_started = time.perf_counter()
            try:
                result = run_case(
                    suite,
                    case,
                    client,
                    progress=progress,
                    verbose=verbose,
                )
            except KeyboardInterrupt as exc:
                spec = estimator_spec(suite.implementation, case.estimator)
                result = _base_result(
                    suite, case, spec.package, _version(spec.package)
                )
                error = _failure(exc, "interrupted")
                result["outcome"] = error
                result["observations"] = [
                    {
                        "id": "measurement-interrupted",
                        "sequence": 0,
                        "role": "measurement",
                        "outcome": error,
                        "extensions": {},
                    }
                ]
                artifact["results"].append(result)
                artifact["run"]["completed_at"] = _now()
                atomic_write(output, artifact)
                if progress is not None:
                    progress(
                        f"[{index}/{total}] {label}: interrupted after "
                        f"{time.perf_counter() - case_started:.3f}s"
                    )
                raise
            artifact["results"].append(result)
            atomic_write(output, artifact)
            if progress is not None:
                progress(
                    f"[{index}/{total}] {label}: "
                    f"{result['outcome']['status']} in "
                    f"{time.perf_counter() - case_started:.3f}s"
                )
    artifact["run"]["completed_at"] = _now()
    atomic_write(output, artifact)
    if progress is not None:
        passed = sum(
            result["outcome"]["status"] == "success"
            for result in artifact["results"]
        )
        failed = len(artifact["results"]) - passed
        progress(
            f"Completed suite {suite.name!r}: {passed} passed, "
            f"{failed} failed in {time.perf_counter() - suite_started:.3f}s; "
            f"artifact: {Path(output).resolve()}"
        )
    return artifact
