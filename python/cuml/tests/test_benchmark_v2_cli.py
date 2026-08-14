# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from cuml.benchmark.cli import default_output_path, main
from cuml.benchmark.suite import load_suite_reference


def test_help_is_suite_aware(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--suite", "cuml_accel", "--help"])
    output = capsys.readouterr().out
    assert "--profile {smoke,standard}" in output
    assert "profile defined by the selected suite" in output


def test_generic_help_does_not_assume_profiles(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    assert "--profile PROFILE" in capsys.readouterr().out


def test_unknown_profile_error_lists_suite_profiles(capsys, tmp_path):
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--suite",
                "cuml_sg",
                "--profile",
                "quick",
                "--output",
                str(tmp_path / "artifact.json"),
            ]
        )
    error = capsys.readouterr().err
    assert (
        "unknown profile 'quick'; available profiles: smoke, standard" in error
    )


def test_default_output_path_uses_suite_profile_and_utc_timestamp(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    suite = load_suite_reference("cuml_sg")
    now = dt.datetime(2026, 8, 13, 17, 30, 12, tzinfo=dt.timezone.utc)
    assert default_output_path(suite, now) == (
        tmp_path / "cuml_sg-standard-20260813T173012Z.json"
    )


def test_default_output_path_does_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    suite = load_suite_reference("cuml_accel", "smoke")
    now = dt.datetime(2026, 8, 13, 17, 30, 12, tzinfo=dt.timezone.utc)
    first = tmp_path / "cuml_accel-smoke-20260813T173012Z.json"
    first.touch()
    assert default_output_path(suite, now) == (
        tmp_path / "cuml_accel-smoke-20260813T173012Z-2.json"
    )


def test_missing_output_generates_and_reports_path(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run_suite(suite, output, argv):
        captured["output"] = Path(output)
        return {"results": []}

    monkeypatch.setattr("cuml.benchmark.harness.run_suite", fake_run_suite)
    assert main(["--suite", "sklearn_cpu", "--profile", "smoke"]) == 0
    output = captured["output"]
    assert output.parent == tmp_path
    assert output.name.startswith("sklearn_cpu-smoke-")
    assert output.suffix == ".json"
    assert f"Writing benchmark artifact to {output}" in capsys.readouterr().out


def test_explicit_output_is_preserved(monkeypatch, tmp_path):
    captured = {}

    def fake_run_suite(suite, output, argv):
        captured["output"] = Path(output)
        return {"results": []}

    monkeypatch.setattr("cuml.benchmark.harness.run_suite", fake_run_suite)
    destination = tmp_path / "chosen.json"
    assert main(["--suite", "sklearn_cpu", "--output", str(destination)]) == 0
    assert captured["output"] == destination.resolve()
