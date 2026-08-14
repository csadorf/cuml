# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for suite-driven neutral-v2 benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path

from .suite import (
    BUILTIN_SUITES,
    SuiteError,
    load_suite_reference,
    suite_profile_names,
)

DEFAULT_SUITE = "cuml_sg"


def _parser(
    profile_names: tuple[str, ...] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cuml.benchmark")
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help=(
            "built-in suite name or strict YAML manifest path; built-ins: "
            + ", ".join(sorted(BUILTIN_SUITES))
            + f" (default: {DEFAULT_SUITE})"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "neutral-v2 JSON artifact path; defaults to a timestamped file "
            "in the current directory"
        ),
    )
    profile_metavar = (
        "{" + ",".join(profile_names) + "}" if profile_names else "PROFILE"
    )
    parser.add_argument(
        "--profile",
        metavar=profile_metavar,
        help="profile defined by the selected suite (default: standard)",
    )
    return parser


def _output_suite_name(suite) -> str:
    path = str(suite.path)
    name = (
        path.removeprefix("builtin:")
        if path.startswith("builtin:")
        else suite.name
    )
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_").lower()
    return slug or "benchmark"


def default_output_path(suite, now: dt.datetime | None = None) -> Path:
    """Return a non-overwriting timestamped artifact path in the current directory."""
    timestamp = (now or dt.datetime.now(dt.timezone.utc)).astimezone(
        dt.timezone.utc
    )
    stem = (
        f"{_output_suite_name(suite)}-{suite.profile_name}-"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    )
    candidate = Path.cwd() / f"{stem}.json"
    suffix = 2
    while candidate.exists():
        candidate = Path.cwd() / f"{stem}-{suffix}.json"
        suffix += 1
    return candidate


def _suite_aware_parser(argv: list[str]) -> argparse.ArgumentParser:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--suite", default=DEFAULT_SUITE)
    probed, _ = probe.parse_known_args(argv)
    try:
        return _parser(suite_profile_names(probed.suite))
    except SuiteError:
        # Full parsing and suite loading below will report the actionable
        # manifest error. Generic help should remain available meanwhile.
        pass
    return _parser()


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
    argument_values = sys.argv[1:] if argv is None else argv
    parser = _suite_aware_parser(argument_values)
    args = parser.parse_args(argument_values)
    try:
        suite = load_suite_reference(args.suite, args.profile)
        _bootstrap_accel_process(suite)
        _prepare_implementation(suite.implementation)
        output = (
            Path(args.output).resolve()
            if args.output
            else default_output_path(suite)
        )
        if args.output is None:
            print(f"Writing benchmark artifact to {output}", flush=True)
        # Import only after accelerator bootstrap/isolation validation.
        from .harness import run_suite

        artifact = run_suite(
            suite,
            output,
            [sys.executable, "-m", "cuml.benchmark", *(argv or sys.argv[1:])],
        )
    except SuiteError as exc:
        parser.error(str(exc))
    return int(
        any(
            result["outcome"]["status"] == "failed"
            for result in artifact["results"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
