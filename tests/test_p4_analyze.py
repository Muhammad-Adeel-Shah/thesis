"""scripts/p4_analyze.py: spec section 5 on the committed scores, reusing P3's analysis.

The reference models' point estimates must reproduce the committed P3 analysis exactly
(same loader, slices, threshold and metric code); a short bootstrap keeps the test quick.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_p4_analyze", ROOT / "scripts" / "p4_analyze.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_p4_analyze"] = module
    spec.loader.exec_module(module)
    return module


A = _module()


def test_section_5_covers_every_row_slice_pool_and_pair() -> None:
    tasks = A.tasks_for((*A.BASELINES, *A.REFERENCE))
    labels = {t[0] for t in tasks}
    assert len(labels) == len(tasks)
    rows = len(A.BASELINES) + len(A.REFERENCE)
    per_row = 2 * len(A.SLICES) + len(A.POOLS) + 2
    pairs = len(A.COMPARATORS) * len(A.BASELINES) * (2 + len(A.SLICES))
    assert len(tasks) == rows * per_row + pairs
    assert len(A.SLICES) == 4 + 6 + 1 and A.SLICES[:4] == A.p3.UNSEEN
    assert set(A.H5_ROWS) == set(A.p3.H5_DECISION)
    assert {"rows_items_1_4", "difference_sign", "tpr_point", "slices"} <= set(A.INTERPRETATIONS)


def test_a_paired_statistic_is_ours_minus_the_baseline() -> None:
    panel = A.p3.Panel({"v": A.np.array([0.0, 1.0])}, {("v", "f"): A.np.array([2.0, 3.0])})
    worse = A.p3.Panel({"v": A.np.array([0.0, 3.0])}, {("v", "f"): A.np.array([1.0, 2.0])})
    stat = A.statistic("auc", ("ours", "base"), ("v", "f"))
    assert stat({"ours": panel, "base": worse}) == 1.0 - 0.5


def test_the_reference_models_reproduce_the_committed_p3_analysis() -> None:
    p3_full = json.loads((ROOT / "reports/p3_results/p3_analysis_full.json").read_text("utf-8"))
    for model in A.REFERENCE:
        panel = A.p3.load_clean(A.LISTS, A.P3_SCORES, model)
        stored = p3_full["descriptive"][model]
        assert A.p3.macro_auc(panel) == stored["macro_auc_unseen"]
        assert A.p3.k0_threshold(panel, "union") == stored["threshold_union"]


def test_every_clean_baseline_row_loads_against_the_committed_lists() -> None:
    sizes = set()
    for row in A.BASELINES:
        panel = A.load_p4(row)
        sizes.add((tuple(sorted((v, len(r)) for v, r in panel.real.items())), len(panel.union)))
    test_reals = 121 + 392 + 2000 + 300          # E3, E4, E5, E7 test reals
    for row in A.H5_ROWS.values():
        assert sum(len(f) for f in A.load_p4_condition(row).fake.values()) == 8655 - test_reals
    assert len(sizes) == 1
