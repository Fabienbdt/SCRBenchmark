"""
Unit tests for utils/statistics.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

from utils.statistics import generate_cld, compute_significance_groups


def _pair_pvalue(pvals, a, b):
    return pvals.get((a, b), pvals.get((b, a), 1.0))


def _shares_letter(cld, a, b):
    return bool(set(cld[a]) & set(cld[b]))


def _assert_cld_constraints(cld, pvals, group_names, alpha=0.05):
    for i, a in enumerate(group_names):
        for b in group_names[i + 1:]:
            p = _pair_pvalue(pvals, a, b)
            shared = _shares_letter(cld, a, b)
            if p < alpha:
                assert not shared, f"{a} and {b} are significant (p={p}) but share '{cld[a]}'/'{cld[b]}'"
            else:
                assert shared, f"{a} and {b} are non-significant (p={p}) but share no letter"


def test_generate_cld_chain_non_significant_bridge():
    """A-B non-significant, B-C non-significant, A-C significant."""
    groups = ["A", "B", "C"]
    pvals = {
        ("A", "B"): 0.50,
        ("B", "C"): 0.50,
        ("A", "C"): 0.001,
    }

    cld = generate_cld(pvals, groups, alpha=0.05)
    _assert_cld_constraints(cld, pvals, groups, alpha=0.05)


def test_generate_cld_two_significant_pairs_and_one_non_significant():
    """A-B significant, A-C significant, B-C non-significant."""
    groups = ["A", "B", "C"]
    pvals = {
        ("A", "B"): 0.001,
        ("A", "C"): 0.001,
        ("B", "C"): 0.50,
    }

    cld = generate_cld(pvals, groups, alpha=0.05)
    _assert_cld_constraints(cld, pvals, groups, alpha=0.05)


def test_compute_significance_groups_filters_non_finite_values():
    data = {
        "algo_a": [0.20, 0.25, np.nan, None],
        "algo_b": [0.70, 0.75, np.inf],
        "algo_c": [0.40, 0.45, "bad_value"],
    }

    cld, global_p, test_name = compute_significance_groups(data, method="nonparametric", alpha=0.05)

    assert set(cld.keys()) == set(data.keys())
    assert np.isfinite(global_p)
    assert isinstance(test_name, str) and len(test_name) > 0
