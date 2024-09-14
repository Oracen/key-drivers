import itertools
from typing import List

import pandas as pd

_DIFF_TAG = "_diff"
_LAG_TAG = "_lag"


def _map_col(col: str, is_diff: bool) -> str:
    return col + (_DIFF_TAG if is_diff else _LAG_TAG)


def decompose_funnel_metrics(
    df: pd.DataFrame,
    date_col: str,
    group_cols: List[str],
    funnel_cols: List[str],
    skip_first_row: bool = True,
) -> pd.DataFrame:
    """Decompose arbitrary funnel metrics into top-level KPI-valued contributions.

    Yields an application of the metric decomposition formula:
        $$
        KPI_t =
            KPI_{t-1} + \\sum_{i=1}^{n} contrib(metric_t)
        $$
    where `contrib(metric_t)` is the contribution of the metric at time `t` to the KPI.

    The return value of this function represents the contributions.

    Args:
        df: DataFrame with columns [date_col, group_cols, metrics]
        date_col: Date column
        group_cols: Grouping columns
        metrics: Metrics to decompose
    Returns:
        DataFrame with columns [date_col, group_cols, metrics, kpi, kpi_contribution]
    """
    df = df.reset_index()  # Ensure no pre-existing index is set

    indexing_cols = group_cols + [date_col]
    all_cols = indexing_cols + funnel_cols

    # Quick preflight check
    df_cols = set(df.columns)
    msg = f"Required columns {all_cols} not found, got {df_cols}"
    assert set(all_cols).issubset(df_cols), msg

    # Build the raw components of the decomposition equation from lagged values (i.e.
    # the metric_{t-1} component) and the diffs (i.e. the del metric_t component)
    df = df[all_cols].set_index(indexing_cols).sort_index()

    df = pd.concat(
        [
            df.groupby(group_cols).shift(1).add_suffix(_LAG_TAG),
            df.groupby(group_cols).diff(1).add_suffix(_DIFF_TAG),
        ],
        axis=1,
    ).fillna(0)

    # Get the cartesian product of all possible combinations of True/False values
    # For our non-target columns below, these will forn the "other" contributions
    # of our n-dimensional prism. The target columm's contribution is always
    # True, as we only consider the diff component and subtract off the lagged
    # value.
    combinations = list(itertools.product(*[(True, False)] * (len(funnel_cols) - 1)))

    # Build each column's total contribution one by one
    output = []
    for target_col in funnel_cols:
        # Mark the first column as the fixed diff, as the element of all lags
        # ends up subtracted off we just need to pin the values we use
        other_cols = [col for col in funnel_cols if col != target_col]
        accumulator = []

        # This loop handles the explosion of terms
        for combo in combinations:
            # Build the contribution of a single direction of the prism
            contrib_cols = [_map_col(target_col, True)] + [
                _map_col(*item) for item in zip(other_cols, combo)
            ]

            # Each subprism is shared amoung all subprisms that contribute delta;
            # rather than make heuristic arguments about assingment priority we
            # just spread the shared delta across all contributing metrics

            # The sharing factor is the reciprocol of the number of diff cols in
            # the equation
            sharing_factor = 1 / (sum(combo) + 1)
            contribution = (
                df[contrib_cols].prod(axis=1) * sharing_factor  # type: ignore
            )

            accumulator.append(contribution)
        # Sum together the contributions
        output.append(pd.concat(accumulator, axis=1).sum(axis=1).rename(target_col))

    start_idx = 1 if skip_first_row else 0
    return pd.concat(output, axis=1)[start_idx:].reset_index()


def validate_funnel_construction(
    df_funnel: pd.DataFrame, funnel_cols: List[str], target: pd.Series
) -> bool:
    error = (df_funnel[funnel_cols].prod(axis=1) - target).abs().max()
    is_close = error < 1e-6
    if not is_close:
        print(f"Large error denotes misspecification; check formula. Error: {error}")
    return is_close


def validate_decomposition(
    df_metrics: pd.DataFrame, metric_cols: List[str], target: pd.Series
) -> bool:
    error = (df_metrics[metric_cols].sum(axis=1) - target).abs().max()
    is_close = error < 1e-6
    if not is_close:
        print(f"Large error denotes misspecification; check formula. Error: {error}")
    return is_close
