# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

from cuml.benchmark.identity import canonical_json, result_id

GOLDEN_ID = (
    "sha256:8cb1f9fa544012b4b8807e81726987ba"
    "331ad2622b2f490817b5f50b628adfcd"
)


def _golden_result():
    return {
        "case_label": "display only",
        "algorithm": "kmeans/Δ",
        "dataset": {
            "name": "blobs-雪",
            "kind": "generated",
            "parameters": {
                "clusters": 8,
                "nested": {"scale": 1e-7, "zero": -0.0},
            },
            "generator": "org.example.gen-v1",
            "fingerprint": None,
            "random_seed": 42,
            "legacy_identity": None,
        },
        "operation": {"name": "fit_predict", "lifecycle": "fit"},
        "input": {
            "dimensions": [
                {"name": "rows", "size": 1000},
                {"name": "features", "size": 32},
            ],
            "data_type": "float32",
            "selection": ["X"],
            "attributes": {"ignored": True},
        },
        "parameters": {
            "declared": {
                "tolerance": 1e-6,
                "labels": ["é", "𝄞"],
                "max_iterations": 100,
            },
            "effective": {"ignored": 1},
        },
    }


def test_rfc8785_golden_vectors_match_dashboard():
    assert canonical_json(
        {"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27]}
    ) == '{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}'
    assert result_id(_golden_result()) == GOLDEN_ID


def test_only_schema_defined_workload_fields_affect_identity():
    original = _golden_result()
    expected = result_id(original)
    for path, replacement in (
        (("algorithm",), "pca"),
        (("dataset", "random_seed"), 43),
        (("operation", "lifecycle"), "inference"),
        (("input", "dimensions"), [{"name": "rows", "size": 1001}]),
        (("input", "selection"), ["y"]),
        (("parameters", "declared"), {"max_iterations": 101}),
    ):
        changed = copy.deepcopy(original)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        assert result_id(changed) != expected

    for path, replacement in (
        (("case_label",), "other"),
        (("input", "attributes"), {"other": True}),
        (("parameters", "effective"), {"other": True}),
        (("implementation",), {"name": "other"}),
        (("observations",), []),
        (("extensions",), {}),
    ):
        changed = copy.deepcopy(original)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        assert result_id(changed) == expected
