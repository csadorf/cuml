# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cuml.benchmark.cli import main


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
