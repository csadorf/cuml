# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for suite-driven neutral-v2 benchmarks."""

from __future__ import annotations

import argparse
import os
import sys

from .suite import SuiteError, load_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cuml.benchmark")
    parser.add_argument(
        "--suite", required=True, help="strict YAML suite manifest"
    )
    parser.add_argument(
        "--output", required=True, help="neutral-v2 JSON artifact path"
    )
    parser.add_argument("--profile", help="suite profile (default: standard)")
    return parser


def _accel_active() -> bool:
    if os.environ.get("CUML_ACCEL_ENABLED") == "1":
        return True
    module = sys.modules.get("cuml.accel")
    return bool(module is not None and module.enabled())


def _prepare_implementation(implementation: str) -> None:
    if implementation == "cuml.accel":
        import cuml.accel

        if not cuml.accel.enabled():
            raise SuiteError(
                "cuml.accel startup bootstrap did not activate the accelerator"
            )
    elif _accel_active():
        raise SuiteError(
            f"cuml.accel is active in isolated {implementation!r} suite"
        )


def _bootstrap_accel_process(suite) -> None:
    if suite.implementation != "cuml.accel" or _accel_active():
        return
    if os.environ.get("CUML_BENCHMARK_ACCEL_BOOTSTRAPPED") == "1":
        raise SuiteError(
            "cuml.accel startup bootstrap did not activate the accelerator"
        )
    # ``python -m cuml.benchmark`` necessarily imports the top-level ``cuml``
    # package before this module. Re-exec once and let cuML's installed .pth
    # hook activate acceleration during interpreter startup, before either
    # cuML or sklearn is imported.
    environment = os.environ.copy()
    environment["CUML_ACCEL_ENABLED"] = "1"
    environment["CUML_BENCHMARK_ACCEL_BOOTSTRAPPED"] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "cuml.benchmark", *sys.argv[1:]],
        environment,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        suite = load_suite(args.suite, args.profile)
        _bootstrap_accel_process(suite)
        _prepare_implementation(suite.implementation)
        # Import only after accelerator bootstrap/isolation validation.
        from .harness import run_suite

        artifact = run_suite(
            suite,
            args.output,
            [sys.executable, "-m", "cuml.benchmark", *(argv or sys.argv[1:])],
        )
    except SuiteError as exc:
        _parser().error(str(exc))
    return int(
        any(
            result["outcome"]["status"] == "failed"
            for result in artifact["results"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
