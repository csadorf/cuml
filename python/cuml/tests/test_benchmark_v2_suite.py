# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from cuml.benchmark.registry import ACCEL_ESTIMATORS, REGISTRIES
from cuml.benchmark.suite import (
    BUILTIN_SUITES,
    SuiteError,
    load_suite,
    load_suite_reference,
)

SUITES = Path(__file__).resolve().parents[1] / "cuml" / "benchmark" / "suites"


def _case_names(path: Path) -> set[str]:
    return {case.estimator for case in load_suite(path).cases}


def test_checked_in_suites_are_strict_and_cover_registries():
    expected = {
        "cuml_sg.yaml": "cuml",
        "cuml_mg.yaml": "cuml.dask",
        "cuml_accel.yaml": "cuml.accel",
        "sklearn_cpu.yaml": "scikit-learn",
    }
    for filename, implementation in expected.items():
        standard = load_suite(SUITES / filename)
        smoke = load_suite(SUITES / filename, "smoke")
        assert standard.implementation == implementation
        assert {c.estimator for c in standard.cases} == set(
            REGISTRIES[implementation]
        )
        assert {c.estimator for c in smoke.cases} == set(
            REGISTRIES[implementation]
        )
        assert [c.id for c in standard.cases] == [
            c.id for c in load_suite(SUITES / filename).cases
        ]


def test_builtin_suites_are_packaged_resources():
    assert BUILTIN_SUITES == {
        "cuml_sg",
        "cuml_mg",
        "cuml_accel",
        "sklearn_cpu",
    }
    for name in BUILTIN_SUITES:
        suite = load_suite_reference(name, "smoke")
        assert suite.path == f"builtin:{name}"
        assert suite.profile_name == "smoke"
        assert suite.cases

        # The optional suffix is accepted as a convenience, while explicit
        # paths remain custom manifests.
        assert load_suite_reference(f"{name}.yaml").name == suite.name


def test_accel_registry_exactly_matches_override_exports():
    overrides = (
        Path(__file__).resolve().parents[1] / "cuml" / "accel" / "_overrides"
    )
    exported = set()
    for path in [
        *sorted((overrides / "sklearn").glob("*.py")),
        overrides / "umap.py",
        overrides / "hdbscan.py",
    ]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                if any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in targets
                ):
                    exported.update(ast.literal_eval(node.value))
    assert set(ACCEL_ESTIMATORS) == exported
    assert _case_names(SUITES / "cuml_accel.yaml") == exported


def test_cpu_coverage_is_comparable_native_accel_union_and_ids_align():
    native = load_suite(SUITES / "cuml_sg.yaml")
    accel = load_suite(SUITES / "cuml_accel.yaml")
    cpu = load_suite(SUITES / "sklearn_cpu.yaml")
    comparable = (
        set(REGISTRIES["cuml"]) | set(REGISTRIES["cuml.accel"])
    ) & set(REGISTRIES["scikit-learn"])
    assert {c.estimator for c in cpu.cases} == comparable
    by_suite = [
        {c.estimator: c for c in suite.cases} for suite in (native, accel, cpu)
    ]
    for estimator in set(by_suite[0]) & set(by_suite[1]) & set(by_suite[2]):
        cases = [mapping[estimator] for mapping in by_suite]
        assert len({c.id for c in cases}) == 1
        assert (
            len(
                {
                    yaml.safe_dump(c.neutral_definition, sort_keys=True)
                    for c in cases
                }
            )
            == 1
        )


def _write(tmp_path: Path, transform=lambda value: value) -> Path:
    value = yaml.safe_load(
        (SUITES / "cuml_accel.yaml").read_text(encoding="utf-8")
    )
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(transform(value), sort_keys=False), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda v: {**v, "mystery": True}, "unknown field"),
        (
            lambda v: {**v, "implementation": "unknown"},
            "invalid implementation",
        ),
        (
            lambda v: {
                **v,
                "cases": [{**v["cases"][0], "operation": "explode"}],
            },
            "invalid operation",
        ),
        (
            lambda v: {**v, "cases": [v["cases"][0], v["cases"][0]]},
            "duplicate case",
        ),
        (lambda v: {**v, "implementation": "cuml.dask"}, "incompatible"),
    ],
)
def test_malformed_suites_are_rejected(tmp_path, mutation, match):
    with pytest.raises(SuiteError, match=match):
        load_suite(_write(tmp_path, mutation))


def test_dispatch_verification_profile_rules(tmp_path):
    assert not load_suite(SUITES / "cuml_accel.yaml").verify_accel_dispatch
    assert load_suite(
        SUITES / "cuml_accel.yaml", "smoke"
    ).verify_accel_dispatch

    def no_warmup(value):
        value["profiles"]["smoke"]["warmups"] = 0
        return value

    with pytest.raises(SuiteError, match="at least one warmup"):
        load_suite(_write(tmp_path, no_warmup), "smoke")

    def standard_profiled(value):
        value["profiles"]["standard"]["verify_accel_dispatch"] = True
        return value

    with pytest.raises(SuiteError, match="standard profile"):
        load_suite(_write(tmp_path, standard_profiled))

    def native_profiled(value):
        value["implementation"] = "cuml"
        value["profiles"]["smoke"]["verify_accel_dispatch"] = True
        value["cases"] = [
            next(c for c in value["cases"] if c["estimator"] == "KMeans")
        ]
        return value

    with pytest.raises(SuiteError, match="only valid"):
        load_suite(_write(tmp_path, native_profiled), "smoke")
