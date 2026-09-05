"""Statistical utilities for clustered biometric evaluation and Q2/Q3 analysis."""

from .bootstrap import (
    cluster_bootstrap_metric,
    paired_cluster_bootstrap_difference,
    percentile_interval,
)
from .inference import holm_adjust, holm_reject, paired_cluster_permutation_test
from .multicriteria import (
    bootstrap_dominance_probability,
    kendall_tau_b,
    non_dominated_mask,
    non_dominated_probability,
    pairwise_rank_reversals,
)

__all__ = [
    "cluster_bootstrap_metric",
    "paired_cluster_bootstrap_difference",
    "percentile_interval",
    "paired_cluster_permutation_test",
    "holm_adjust",
    "holm_reject",
    "non_dominated_mask",
    "bootstrap_dominance_probability",
    "non_dominated_probability",
    "kendall_tau_b",
    "pairwise_rank_reversals",
]
