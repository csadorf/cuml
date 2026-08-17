# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from cuml.benchmark.harness import _base_result
from cuml.benchmark.registry import ACCEL_ESTIMATORS, REGISTRIES
from cuml.benchmark.suite import (
    BUILTIN_SUITES,
    SuiteError,
    load_suite,
    load_suite_reference,
    suite_profile_names,
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


def test_suite_profile_names_supports_builtins_and_custom_paths(tmp_path):
    assert suite_profile_names("cuml_accel") == ("smoke", "standard")
    custom = _write(tmp_path)
    assert suite_profile_names(custom) == ("smoke", "standard")


def test_unknown_profile_reports_available_profiles():
    with pytest.raises(
        SuiteError,
        match=("unknown profile 'quick'; available profiles: smoke, standard"),
    ):
        load_suite_reference("cuml_sg", "quick")


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


def test_comparable_cases_align_across_all_builtin_suites():
    native = load_suite(SUITES / "cuml_sg.yaml")
    dask = load_suite(SUITES / "cuml_mg.yaml")
    accel = load_suite(SUITES / "cuml_accel.yaml")
    cpu = load_suite(SUITES / "sklearn_cpu.yaml")
    comparable = (
        set(REGISTRIES["cuml"]) | set(REGISTRIES["cuml.accel"])
    ) & set(REGISTRIES["scikit-learn"])
    assert {c.estimator for c in cpu.cases} == comparable
    by_suite = [
        {c.estimator: c for c in suite.cases}
        for suite in (native, dask, accel, cpu)
    ]
    for estimator in set().union(*by_suite):
        cases = [
            mapping[estimator] for mapping in by_suite if estimator in mapping
        ]
        if len(cases) < 2:
            continue
        assert len({c.id for c in cases}) == 1
        assert len(
            {
                _base_result(suite, mapping[estimator], "test", "1")["id"]
                for suite, mapping in zip(
                    (native, dask, accel, cpu), by_suite, strict=True
                )
                if estimator in mapping
            }
        ) == 1
        assert (
            len(
                {
                    yaml.safe_dump(c.neutral_definition, sort_keys=True)
                    for c in cases
                }
            )
            == 1
        )


def test_standard_shapes_follow_established_benchmark_workloads():
    cases = {
        case.estimator: case
        for case in load_suite(SUITES / "cuml_sg.yaml").cases
    }
    assert (
        cases["LogisticRegression"].rows,
        cases["LogisticRegression"].features,
    ) == (
        10_500_000,
        128,
    )
    assert (cases["KernelRidge"].rows, cases["KernelRidge"].features) == (
        1_000,
        64,
    )
    assert (
        cases["StandardScaler"].rows,
        cases["StandardScaler"].features,
    ) == (
        52_000,
        512,
    )
    assert (cases["UMAP"].rows, cases["UMAP"].features) == (10_000, 100)


def test_smoke_profiles_scale_standard_rows_without_changing_features():
    for name in BUILTIN_SUITES:
        standard = load_suite_reference(name)
        smoke = load_suite_reference(name, "smoke")
        standard_cases = {case.estimator: case for case in standard.cases}
        for case in smoke.cases:
            reference = standard_cases[case.estimator]
            assert case.rows == max(32, round(reference.rows * 0.01))
            assert case.features == reference.features


def test_scaled_workloads_have_distinct_result_ids():
    standard = load_suite_reference("cuml_sg")
    smoke = load_suite_reference("cuml_sg", "smoke")
    standard_case = standard.cases[0]
    smoke_case = smoke.cases[0]
    assert _base_result(standard, standard_case, "cuml", "1")["id"] != (
        _base_result(smoke, smoke_case, "cuml", "1")["id"]
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
