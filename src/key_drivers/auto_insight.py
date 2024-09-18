import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
from sklearn import cluster

_DRIVER_COL_NAMES = {"sum": "total", "count": "vcount", "nunique": "id_count"}


@dataclass
class DrivingFactor:
    categories: Dict[str, str]
    total: float
    vcount: int
    id_count: int


def _gen_prefix(col: str):
    return col + "_"


def _select_top_drivers(
    df: pd.DataFrame,
    threshold: float,
    norm_total: float,
    id_col: str,
    group_cols: List[List[str]],
    target_col: str,
    top_only=False,
) -> Tuple[pd.DataFrame, List[DrivingFactor]]:
    """Select the top driving factors of our target for a combination of dimensions

    Args:
        df: DataFrame with columns [group_cols, target_col]
        threshold: Minimum contribution to consider
        norm_total: Normalisation term for the target column
        group_cols: List of columns to group by
        target_col: Target column
        top_only: If True, only the top driver is returned

    Returns:
        Tuple of (DataFrame, List[Dict[str, Any]]) where the DataFrame is the input
        DataFrame with the selected rows removed, and the list is the driving factors
        found by the search
    """
    # Build all combos at once, so we're fairly judging multicats
    constructor = []
    for cols in group_cols:
        df_values = (
            df.groupby(cols)
            .agg({target_col: ["sum", "count"], id_col: "nunique"})
            .rename(columns=_DRIVER_COL_NAMES)
        )
        df_values.columns = df_values.columns.droplevel(0)

        df_values["total"] /= norm_total
        df_values["total_abs"] = df_values["total"].abs()
        constructor.append(df_values.reset_index())

    # Do all the comparisons at once
    df_values = pd.concat(constructor).reset_index(drop=True)

    # We want to grab the smallest chunks with the biggest impacts, then ditch abs
    df_values = df_values[df_values.total_abs >= threshold].sort_values("id_count")
    df_values.pop("total_abs")
    if len(df_values) == 0:
        return df, []

    if top_only:
        df_values = df_values.head(1)

    # Incredibly ugly filtering logic to do everything in a single pass
    output = []
    outer_filter = []
    for row in df_values.to_dict(orient="records"):
        item = {}
        categories = {}

        # The inner filter will be "AND" to ID the row which gen'd the item
        inner_filter = []
        for key, value in row.items():
            if value != value or value is None:
                # NaNs and None values are not allowed, use this check to get rid
                # of empty keys
                continue
            if key in set(_DRIVER_COL_NAMES.values()):
                item[key] = value
                continue

            # Store the category and mark the filter for removal
            categories[str(key)] = str(value)
            inner_filter.append(df[key] == value)

        # Store the output  + filter
        output.append(
            DrivingFactor(
                categories=categories,
                total=item["total"],
                vcount=item["vcount"],
                id_count=item["id_count"],
            )
        )
        outer_filter.append(pd.concat(inner_filter, axis=1).all(axis=1))

    # Finally, we want to filter all ID'd rows
    filter_idx = pd.concat(outer_filter, axis=1).any(axis=1)

    # We want to remove the entries we've selected from the dataframe
    return df[~filter_idx], output


def find_key_drivers(
    df: pd.DataFrame,
    target_groupings: int,
    id_col: str,
    group_cols: List[str],
    funnel_cols: List[str],
    target_threshold: Optional[float] = None,
    use_id_col: bool = True,
    target_col_name: str = "target",
    copy: bool = True,
) -> List[DrivingFactor]:
    """
    Find the top drivers of a target column in a dataframe, using a combination of
    groupings to identify the key drivers.

    Args:
        df: DataFrame with columns [group_cols, funnel_cols, target_col]
        target_groupings: Number of top drivers to find
        id_col: Individual ID column
        group_cols: List of columns to group by
        funnel_cols: List of columns to consider for the funnel
        target_threshold: Minimum contribution to consider
        use_id_col: If True, individual IDs are considered
        target_col_name: Name of the target column
        copy: If True, do not mutate the input DataFrame

    Returns:
        List of DrivingFactor objects representing the top drivers
    """
    # Make sure we're not editing the base dataframe
    if copy:
        df = df.copy()

    # Total values gives us a stopping criterion to bundle things into an "other" group
    total_values = target_groupings

    # We want to ensure that we have a minimum threshold for each group
    threshold = (
        target_threshold if target_threshold is not None else 1 / target_groupings
    )

    # Set up the totals for sorting, pre-compute the normalisation term
    df[target_col_name] = df[funnel_cols].sum(axis=1)
    norm_total = df[target_col_name].sum()

    low_end = 0
    combinations = [
        list(map(list, itertools.combinations(group_cols, selections)))
        for selections in range(len(group_cols), 0, -1)
    ]

    # If we're checking for individual observation impact, we want to do that first.
    # Huge individuals can skew the results of the other groupings so get rid of them
    # first
    if use_id_col:
        low_end += 1
        combinations = [[[id_col]]] + combinations

    # Begin scanning through the combinations, stopping when we've hit the target.
    # Some of this is hideously inefficient, but the inefficiency is about the only
    # way to guarantee that we're not mis-allocating big contributors. It's my
    # understanding that this is max-min-sum problem, which is NP-hard, so we're
    # stuck with ugly brute-forcing. There's probably a more elegant way, but
    # readability is more important than efficiency here.
    top_drivers = []
    for idx, combo in enumerate(combinations):
        # The "all cats" and "one-by-one" case is immune to issues related to
        # multi-set membership, so we skip them. We also skip the case where we
        # group by individual ID, and thus skew that forward
        multilabel_possible = not (idx <= low_end or idx == len(combinations) - 1)
        while True:
            # Use this loop to do multilabels one by one, in combo with the above bool
            df, output = _select_top_drivers(
                df,
                threshold,
                norm_total,
                id_col,
                combo,
                target_col_name,
                top_only=multilabel_possible,
            )
            top_drivers.extend(output)
            total_values -= len(output)

            if not multilabel_possible or len(output) == 0:
                # If we're doing "all at once" or we've run out of big contributors,
                # break out of the inner while
                break

        # If we have enough values, we can break out of the outer loop
        if total_values <= 0:
            return top_drivers

    return top_drivers


def reduce_cat_columns(
    df_joined: pd.DataFrame,
    id_col: str,
    cat_cols: List[str],
    target_cols: List[str],
    member_pct: float = 0.75,
    copy: bool = True,
):
    """Reduce the number of categories in a DataFrame by clustering similar behaviours

    Args:
        df_joined: DataFrame with columns [cat_cols, target_cols]
        cat_cols: List of columns to reduce
        target_cols: List of target columns
        member_pct: Minimum proportion of a category in a cluster to consider
        copy: If True, do not mutate the input DataFrame

    Returns:
        DataFrame with reduced categories
    """
    # Prevent mutation of the input
    if copy:
        df_joined = df_joined.copy()

    # Basic idea is we want to lump together the categories that behave similarly.
    # We'll use UMAP to do this, and then use DBSCAN to find the clusters. This is
    # a bit of a hack, but it's a good way to get a rough idea of what's going on
    # as DBSCAN's density basis can work around the quirks of UMAP's topological
    # clustering
    df_cat_cols = pd.concat(
        [
            pd.get_dummies(df_joined[col].astype("category").cat.codes).add_prefix(
                _gen_prefix(col)
            )
            for col in cat_cols
        ],
        axis=1,
    )

    # We want to group by the ID column, so we can get the mean of the categories; this
    # allows basic support for type-2 SCD derived data
    df_cat_cols = df_cat_cols.groupby(df_joined[id_col]).mean()

    target = df_joined[target_cols].groupby(df_joined[id_col]).mean()

    embedding = umap.UMAP(
        # DICE works with partial binary data
        metric="dice",
        target_metric="l1",
        # We want this to be mostly about the target, but not entirely. The categories
        # should pull together, but with their behaviour dominated by the target
        target_weight=0.75,
        # Let things collapse right down
        min_dist=0.01,
        spread=1.0,
        # Favour local features
        n_neighbors=9,
        # Give some room for the categories to breathe
        n_components=len(target_cols),
        set_op_mix_ratio=0.66,
    )

    # Transform the data, then convince type checking our output is correct
    # Ignoring the userwarning, we have no control over scipy anway
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".* inverse_transform .*")
        feat_tx_arr = embedding.fit_transform(df_cat_cols, y=target)
    assert isinstance(feat_tx_arr, np.ndarray)

    # Style points only; again this calms down the type checking
    feat_tx = pd.DataFrame(
        feat_tx_arr,
        columns=[f"cmp_{idx}" for idx in range(feat_tx_arr.shape[1])],
        index=target.index,
    )

    # Very local DB scan, we want a high-ish number of clusters
    dbscan = cluster.DBSCAN(min_samples=3)
    clusers_base = dbscan.fit_predict(feat_tx).tolist()
    clusters = pd.Series(
        df_joined[id_col].map(dict(zip(target.index, clusers_base))),
        index=df_joined.index,
        name="auto_cluster",
    )

    # Where large proportions of a few categories are in a single cluster, we'll
    # lump them together. This is a heuristic argument but tends to produce good
    # clumps without too much "black magic" per normal clusters
    for col in cat_cols:
        props = (
            df_joined[col].groupby(clusters).value_counts()
            / df_joined[col].value_counts()
        )
        # Drop the noise cluster, it's not defined by a set of common behaviours
        # and will just confuse things
        props.drop(-1, inplace=True, errors="ignore")

        # If it has member_pct% of the data in a single cluster we deem this a common
        # set of behaviours
        props = props[props > member_pct].reset_index()
        if len(props) == 0:
            # No common behaviours, skip
            continue

        # Create new categories and update the old values on a per-cluster basis
        for item in props.auto_cluster.unique():
            tmp = props[props.auto_cluster == item]
            if len(tmp) < 2:
                # Skip singletons
                continue

            merged_cat = "|".join(tmp[col].astype(str))
            merged_cat = f"({merged_cat})"

            df_joined[col] = df_joined[col].replace(tmp[col].to_list(), merged_cat)

    return df_joined
