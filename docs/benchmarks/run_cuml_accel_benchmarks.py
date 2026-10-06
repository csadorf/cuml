#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the accel performance suite and export fresh documentation measurements."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from generate_cuml_accel_benchmarks import validate_publication_data

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "python/cuml/cuml/benchmark/suites/accel-performance.yaml"
EXTENSION = "com.nvidia.cuml.benchmark"
ACCEL_EXTENSION = "com.nvidia.cuml.accel"
PACKAGES = {"cuml", "scikit-learn", "hdbscan", "umap-learn"}


def collect_system() -> dict:
    """Capture Linux CPU/RAM and the single CUDA-visible benchmark GPU."""
    import cupy

    cpu_info = Path("/proc/cpuinfo").read_text()
    model = next(
        line.split(":", 1)[1].strip()
        for line in cpu_info.splitlines()
        if line.startswith("model name")
    )
    cores = set()
    logical = 0
    for block in cpu_info.strip().split("\n\n"):
        fields = dict(
            line.split(":", 1) for line in block.splitlines() if ":" in line
        )
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "processor" in fields:
            logical += 1
            if "physical id" not in fields or "core id" not in fields:
                raise ValueError(
                    "CPU physical/core IDs unavailable in /proc/cpuinfo"
                )
            cores.add((fields["physical id"], fields["core id"]))
    memory = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    if cupy.cuda.runtime.getDeviceCount() != 1:
        raise ValueError(
            "Select exactly one visible GPU using CUDA_VISIBLE_DEVICES"
        )
    props = cupy.cuda.runtime.getDeviceProperties(0)
    gpu_name = props["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    return {
        "components": [
            {
                "type": "cpu",
                "name": model,
                "count": 1,
                "attributes": {
                    "logical_cores": logical,
                    "physical_cores": len(cores),
                },
            },
            {
                "type": "memory",
                "name": "System memory",
                "count": 1,
                "attributes": {"total_memory_bytes": memory},
            },
            {
                "type": "gpu",
                "name": gpu_name,
                "count": 1,
                "attributes": {
                    "total_memory_bytes": int(props["totalGlobalMem"])
                },
            },
        ]
    }


def package_versions(artifact: dict) -> dict:
    """Extract recorded versions, rejecting absent or unknown packages."""
    packages = artifact["run"]["software"]["packages"]
    names = [package["name"] for package in packages]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate package metadata")
    versions = {
        p["name"]: p["version"] for p in packages if p["name"] in PACKAGES
    }
    if not versions or any(
        not version or version == "unknown" for version in versions.values()
    ):
        raise ValueError(
            "Artifacts must record known publication package versions"
        )
    return versions


def median_time(result: dict, repetitions: int) -> float:
    """Require a completed case and exactly the expected wall measurements."""
    if result["outcome"]["status"] != "success":
        raise ValueError(f"Failed case: {result['case_label']}")
    observations = [
        o for o in result["observations"] if o["role"] == "measurement"
    ]
    if (
        len(observations) != repetitions
        or len({o["id"] for o in observations}) != repetitions
    ):
        raise ValueError(f"Incomplete measurements: {result['case_label']}")
    values = []
    for observation in observations:
        timings = [
            t["value"]
            for t in observation["timings"]
            if t["name"] == "wall_time" and t["unit"] == "s"
        ]
        if observation["outcome"]["status"] != "success" or len(timings) != 1:
            raise ValueError("Failed or ambiguous measurement")
        value = timings[0]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Wall times must be finite and positive")
        values.append(value)
    return statistics.median(values)


def verify_dispatch(result: dict, warmups: int) -> None:
    """Require successful warmup evidence with GPU calls and no CPU fallback."""
    observations = [o for o in result["observations"] if o["role"] == "warmup"]
    if (
        len(observations) != warmups
        or len({o["id"] for o in observations}) != warmups
    ):
        raise ValueError("Missing accel warmup verification")
    for observation in observations:
        evidence = observation.get("extensions", {}).get(ACCEL_EXTENSION, {})
        if (
            observation["outcome"]["status"] != "success"
            or evidence.get("gpu_calls", 0) <= 0
            or evidence.get("cpu_calls") != 0
        ):
            raise ValueError(
                f"Invalid GPU dispatch evidence: {result['case_label']}"
            )


def export_data(cpu: dict, gpu: dict, system: dict, plan) -> dict:
    """Pair validated observations and produce the documentation contract."""
    import jsonschema
    from cuml.benchmark.identity import result_id
    from cuml.benchmark.schemas import benchmark_result_schema

    validator = jsonschema.Draft202012Validator(
        json.loads(benchmark_result_schema().read_text()),
        format_checker=jsonschema.FormatChecker(),
    )
    for artifact in (cpu, gpu):
        errors = list(validator.iter_errors(artifact))
        if errors:
            raise ValueError(
                f"Invalid benchmark artifact: {errors[0].message}"
            )

    cases = plan.runs[0].cases
    expected = {case.id: case for case in cases}
    if len(expected) != 168 or len({case.label for case in cases}) != 168:
        raise ValueError(
            "Publication suite must contain 168 unique labeled workloads"
        )
    if cpu["schema_version"] != 2 or gpu["schema_version"] != 2:
        raise ValueError("Expected benchmark artifact schema version 2")
    cpu_versions, gpu_versions = package_versions(cpu), package_versions(gpu)
    if any(
        cpu_versions[name] != gpu_versions[name]
        for name in cpu_versions.keys() & gpu_versions.keys()
    ):
        raise ValueError("CPU and GPU package versions differ")
    versions = {**cpu_versions, **gpu_versions}
    if set(versions) != PACKAGES:
        raise ValueError(
            "Artifacts must record all four publication package versions"
        )
    if cpu["run"]["system"] != gpu["run"]["system"]:
        raise ValueError("CPU and GPU systems differ")
    if cpu["run"]["methodology"] != gpu["run"]["methodology"]:
        raise ValueError("CPU and GPU methodologies differ")
    indexes = []
    for artifact, backend in ((cpu, "scikit-learn"), (gpu, "cuml.accel")):
        execution = artifact["run"]["extensions"][EXTENSION]
        if (
            execution["implementation"] != backend
            or execution["profile"] != "standard"
        ):
            raise ValueError("Unexpected backend or profile")
        requested = {
            entry["case_id"]: entry for entry in execution["execution_plan"]
        }
        if len(requested) != len(execution["execution_plan"]) or set(
            requested
        ) != set(expected):
            raise ValueError("Execution plan does not match publication suite")
        index = {}
        for result in artifact["results"]:
            key = result["id"]
            if key in index or key not in expected or result_id(result) != key:
                raise ValueError(
                    "Duplicate, unexpected, or invalid workload ID"
                )
            case = expected[key]
            entry = requested[key]
            if (
                result["case_label"] != case.label
                or entry["case_label"] != case.label
                or entry["warmups"] != case.warmups
                or entry["repetitions"] != case.repetitions
                or entry["timeout_sec"] != case.timeout_sec
            ):
                raise ValueError(
                    "Result labels or execution settings do not match suite"
                )
            index[key] = result
        if set(index) != set(expected):
            raise ValueError("Missing publication cases")
        indexes.append(index)
    records = []
    for case in sorted(cases, key=lambda case: case.label):
        cpu_result, gpu_result = (index[case.id] for index in indexes)
        gpu_time = median_time(gpu_result, case.repetitions)
        verify_dispatch(gpu_result, case.warmups)
        outcome = cpu_result["outcome"]
        timed_out = (
            outcome["status"] == "failed"
            and outcome.get("error", {}).get("type") == "TimeoutExpired"
        )
        if timed_out:
            raise ValueError(
                f"CPU case {case.label!r} timed out: complete-case deadlines "
                "cannot establish operation-time speedup bounds"
            )
        record = {
            "case_label": case.label,
            "cpu_median_sec": median_time(cpu_result, case.repetitions),
            "gpu_median_sec": gpu_time,
            "rows": case.measured_rows,
            "features": case.features,
        }
        if case.estimator == "PCA" and ".rank" not in case.label:
            record["components"] = case.parameters["n_components"]
        records.append(record)
    return validate_publication_data(
        {
            "schema_version": 1,
            "system": system,
            "packages": versions,
            "records": records,
        }
    )


def main(argv=None) -> int:
    """Run both backends or export their existing artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if os.environ.get("CUML_ACCEL_ENABLED") == "1":
            raise ValueError(
                "Run the driver without CUML_ACCEL_ENABLED=1; it starts isolated backends"
            )
        from cuml.benchmark._utils import atomic_write
        from cuml.benchmark.suite import load_suite

        plan = load_suite(SUITE, "standard")
        directory = args.output_dir.resolve()
        snapshot = directory / "system.json"
        from cuml.benchmark.harness import _run_record

        if args.resume or args.export_only:
            metadata = json.loads(snapshot.read_text())
            system = metadata["publication_system"]
            if args.resume and collect_system() != system:
                raise ValueError(
                    "Resume system differs from saved system metadata"
                )
        else:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(
                    "Output directory must be empty; use --resume or --export-only"
                )
            system = collect_system()
            metadata = {
                "publication_system": system,
                "benchmark_system": _run_record(plan.runs[0], [])["system"],
            }
            atomic_write(snapshot, metadata)
        if args.resume:
            from cuml.benchmark.harness import _validate_resume_artifact

            # Reject incompatibility before workers can return aggregate exit 1
            # while leaving previously complete artifacts untouched.
            for run in plan.runs:
                artifact_path = directory / f"{run.implementation}.json"
                if artifact_path.exists():
                    previous = json.loads(artifact_path.read_text())
                    current = {
                        "schema_version": 2,
                        "run": _run_record(run, []),
                    }
                    _validate_resume_artifact(
                        previous, current, {case.id for case in run.cases}
                    )
        if not args.export_only:
            command = [
                sys.executable,
                "-m",
                "cuml.benchmark",
                "--suite",
                str(SUITE),
                "--profile",
                "standard",
                "--implementation",
                "scikit-learn",
                "--implementation",
                "cuml.accel",
                "--output",
                str(directory),
            ]
            if args.resume:
                command.append("--resume")
            status = subprocess.run(command, cwd=ROOT, check=False).returncode
            if status != 0:
                raise ValueError(
                    f"Benchmark process exited with unexpected status {status}"
                )
            if collect_system() != system:
                raise ValueError("System metadata changed during benchmarking")
        cpu = json.loads((directory / "scikit-learn.json").read_text())
        gpu = json.loads((directory / "cuml.accel.json").read_text())
        if any(
            artifact["run"]["system"] != metadata["benchmark_system"]
            for artifact in (cpu, gpu)
        ):
            raise ValueError(
                "Artifacts do not match the saved system snapshot"
            )
        data = export_data(cpu, gpu, system, plan)
        destination = directory / "benchmark-data.json"
        atomic_write(destination, data)
        print(f"Wrote {destination}")
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        StopIteration,
    ) as exc:
        print(f"Unable to produce publication data: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
