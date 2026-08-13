# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict suite parsing and neutral case resolution."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .registry import REGISTRIES


class SuiteError(ValueError):
    pass


TOP_FIELDS = frozenset(
    {"version", "name", "implementation", "profiles", "cases"}
)
PROFILE_FIELDS = frozenset(
    {"warmups", "repetitions", "size_scale", "verify_accel_dispatch"}
)
CASE_FIELDS = frozenset(
    {"estimator", "dataset", "operation", "shape", "parameters"}
)
SHAPE_FIELDS = frozenset({"rows", "features"})
OPERATIONS = frozenset(
    {
        "fit",
        "fit_predict",
        "fit_transform",
        "predict",
        "transform",
        "kneighbors",
        "score_samples",
    }
)
IMPLEMENTATIONS = frozenset(REGISTRIES)
BUILTIN_SUITES = frozenset({"cuml_sg", "cuml_mg", "cuml_accel", "sklearn_cpu"})


def _unknown(
    value: dict[str, Any], allowed: frozenset[str], where: str
) -> None:
    fields = set(value) - allowed
    if fields:
        raise SuiteError(
            f"unknown field(s) in {where}: {', '.join(sorted(fields))}"
        )


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuiteError(f"{where} must be a mapping")
    return value


@dataclass(frozen=True)
class ResolvedCase:
    estimator: str
    dataset: str
    operation: str
    rows: int
    features: int
    parameters: dict[str, Any]
    warmups: int
    repetitions: int

    @property
    def neutral_definition(self) -> dict[str, Any]:
        return {
            "algorithm": self.estimator,
            "dataset": self.dataset,
            "operation": self.operation,
            "shape": {"rows": self.rows, "features": self.features},
            "parameters": self.parameters,
        }

    @property
    def id(self) -> str:
        encoded = json.dumps(
            self.neutral_definition, sort_keys=True, separators=(",", ":")
        )
        return f"case-{hashlib.sha256(encoded.encode()).hexdigest()[:20]}"


@dataclass(frozen=True)
class Suite:
    path: str | Path
    name: str
    implementation: str
    profile_name: str
    verify_accel_dispatch: bool
    cases: tuple[ResolvedCase, ...]


def load_suite(path: str | Path, profile: str | None = None) -> Suite:
    suite_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteError(f"unable to load suite {suite_path}: {exc}") from exc
    raw = _mapping(raw, "suite")
    _unknown(raw, TOP_FIELDS, "suite")
    missing = TOP_FIELDS - set(raw)
    if missing:
        raise SuiteError(
            f"missing suite field(s): {', '.join(sorted(missing))}"
        )
    if raw["version"] != 2:
        raise SuiteError("suite version must be 2")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise SuiteError("suite name must be a non-empty string")
    implementation = raw["implementation"]
    if implementation not in IMPLEMENTATIONS:
        raise SuiteError(f"invalid implementation {implementation!r}")
    profiles = _mapping(raw["profiles"], "profiles")
    profile_name = profile or "standard"
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise SuiteError(
            f"unknown profile {profile_name!r}; available profiles: {available}"
        )
    profile_data = _mapping(
        profiles[profile_name], f"profile {profile_name!r}"
    )
    _unknown(profile_data, PROFILE_FIELDS, f"profile {profile_name!r}")
    for required in ("warmups", "repetitions", "size_scale"):
        if required not in profile_data:
            raise SuiteError(
                f"profile {profile_name!r} is missing {required!r}"
            )
    warmups, repetitions = profile_data["warmups"], profile_data["repetitions"]
    scale = profile_data["size_scale"]
    if not isinstance(warmups, int) or warmups < 0:
        raise SuiteError("warmups must be a non-negative integer")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise SuiteError("repetitions must be a positive integer")
    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or scale <= 0
        or scale > 1
    ):
        raise SuiteError("size_scale must be in (0, 1]")
    verify = profile_data.get("verify_accel_dispatch", False)
    if not isinstance(verify, bool):
        raise SuiteError("verify_accel_dispatch must be boolean")
    if verify and implementation != "cuml.accel":
        raise SuiteError(
            "verify_accel_dispatch is only valid for cuml.accel suites"
        )
    if verify and warmups < 1:
        raise SuiteError("verify_accel_dispatch requires at least one warmup")
    if (
        implementation == "cuml.accel"
        and profile_name == "standard"
        and verify
    ):
        raise SuiteError("the accel standard profile must not verify dispatch")

    raw_cases = raw["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError("cases must be a non-empty list")
    resolved = []
    seen = set()
    registry = REGISTRIES[implementation]
    for index, item in enumerate(raw_cases):
        where = f"case {index}"
        item = _mapping(item, where)
        _unknown(item, CASE_FIELDS, where)
        missing = CASE_FIELDS - set(item)
        if missing:
            raise SuiteError(
                f"{where} is missing field(s): {', '.join(sorted(missing))}"
            )
        estimator = item["estimator"]
        if estimator not in registry:
            raise SuiteError(
                f"{where}: estimator {estimator!r} is incompatible with {implementation!r}"
            )
        if item["operation"] not in OPERATIONS:
            raise SuiteError(
                f"{where}: invalid operation {item['operation']!r}"
            )
        if not isinstance(item["dataset"], str) or not item["dataset"]:
            raise SuiteError(f"{where}: dataset must be a non-empty string")
        shape = _mapping(item["shape"], f"{where} shape")
        _unknown(shape, SHAPE_FIELDS, f"{where} shape")
        if set(shape) != SHAPE_FIELDS or any(
            not isinstance(shape[k], int) or shape[k] < 1 for k in SHAPE_FIELDS
        ):
            raise SuiteError(
                f"{where}: shape requires positive integer rows and features"
            )
        parameters = _mapping(item["parameters"], f"{where} parameters")
        rows = max(32, round(shape["rows"] * scale))
        # Profiles shrink observations while preserving feature-dependent
        # estimator parameters (for example PCA's n_components).
        features = shape["features"]
        case = ResolvedCase(
            estimator,
            item["dataset"],
            item["operation"],
            rows,
            features,
            json.loads(json.dumps(parameters)),
            warmups,
            repetitions,
        )
        if case.id in seen:
            raise SuiteError(f"duplicate case identity {case.id!r}")
        seen.add(case.id)
        resolved.append(case)
    return Suite(
        suite_path,
        raw["name"],
        implementation,
        profile_name,
        verify,
        tuple(resolved),
    )


@contextmanager
def _suite_reference_path(reference: str | Path):
    reference_text = str(reference)
    builtin_name = reference_text.removesuffix(".yaml")
    if builtin_name in BUILTIN_SUITES and Path(reference_text).parent == Path(
        "."
    ):
        resource = importlib.resources.files("cuml.benchmark.suites").joinpath(
            f"{builtin_name}.yaml"
        )
        with importlib.resources.as_file(resource) as resource_path:
            yield resource_path, builtin_name
    else:
        yield Path(reference).resolve(), None


def suite_profile_names(reference: str | Path) -> tuple[str, ...]:
    """Return sorted profile names without resolving or executing cases."""
    with _suite_reference_path(reference) as (suite_path, _):
        try:
            raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SuiteError(
                f"unable to load suite {suite_path}: {exc}"
            ) from exc
    raw = _mapping(raw, "suite")
    profiles = _mapping(raw.get("profiles"), "profiles")
    if not profiles:
        raise SuiteError("profiles must be a non-empty mapping")
    return tuple(sorted(profiles))


def load_suite_reference(
    reference: str | Path, profile: str | None = None
) -> Suite:
    """Load a packaged built-in suite by name or a custom suite by path."""
    with _suite_reference_path(reference) as (suite_path, builtin_name):
        suite = load_suite(suite_path, profile)
    if builtin_name is not None:
        return replace(suite, path=f"builtin:{builtin_name}")
    return suite
