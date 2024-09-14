import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

URL = (
    "https://www.kaggle.com/competitions/"
    "store-sales-time-series-forecasting/data?select=stores.csv"
)

BASE_PRICES = {
    "AUTOMOTIVE": 28,
    "BABY CARE": 9,
    "BEAUTY": 12,
    "BEVERAGES": 5,
    "BOOKS": 15,
    "BREAD/BAKERY": 6,
    "CELEBRATION": 10,
    "CLEANING": 5,
    "DAIRY": 3,
    "DELI": 8,
    "EGGS": 3,
    "FROZEN FOODS": 5,
    "GROCERY I": 4,
    "GROCERY II": 6,
    "HARDWARE": 20,
    "HOME AND KITCHEN I": 9,
    "HOME AND KITCHEN II": 16,
    "HOME APPLIANCES": 50,
    "HOME CARE": 5,
    "LADIESWEAR": 20,
    "LAWN AND GARDEN": 18,
    "LINGERIE": 25,
    "LIQUOR,WINE,BEER": 25,
    "MAGAZINES": 5,
    "MEATS": 13,
    "PERSONAL CARE": 4,
    "PET SUPPLIES": 10,
    "PLAYERS AND ELECTRONICS": 47,
    "POULTRY": 8,
    "PREPARED FOODS": 8,
    "PRODUCE": 1.5,
    "SCHOOL AND OFFICE SUPPLIES": 11,
    "SEAFOOD": 17,
}


@dataclass
class DemoData:
    df_sales: pd.DataFrame
    df_stores: pd.DataFrame
    group_variables: List[str]
    funnel_variables: List[str]


# Simulate some more dynamic data
def promotions_to_profit_discount_factor(promotions, avg_price):
    discount_factor = np.log2(promotions + 1) / 100
    rarity = np.log2(avg_price + 1)
    return discount_factor * rarity


# Add some quick additional context - apologies if the pandas-fu is a bit wonky.
# Long and short of it is we want how many months since dataset begin we start
# seeing data


def categorise_openings(months_since_open: int) -> str:
    # I know this won't age well on a dynamic dataset, sue me
    if months_since_open == 0:
        return "already open"
    if months_since_open < 24:
        return "first_24"
    if months_since_open < 36:
        return "rush_open"
    return "brand_new"


def load_demo_data(
    path: str, avg_price_override: Optional[Dict[str, float]] = None
) -> DemoData:

    base_path = pathlib.Path(path)

    if not (base_path / "train.csv").is_file():
        raise FileNotFoundError(
            f"Missing train.csv, be sure you've downloaded the dataset. See {URL} "
            "to download these files."
        )

    # Average price of an item purchased in each department
    avg_prices = {**BASE_PRICES, **(avg_price_override or {})}

    # Load in data and do some basic conversions
    df_sales = pd.read_csv(base_path / "train.csv")
    df_transactions = pd.read_csv(base_path / "transactions.csv")
    df_stores = pd.read_csv(base_path / "stores.csv")
    df_stores.columns = [
        item if item.startswith("store") else "store_" + item
        for item in df_stores.columns
    ]

    for item in [df_sales, df_transactions]:
        item["date"] = pd.to_datetime(item.date)

    # Based on the information provided, develop our top-level KPI
    base_columns = ["sales", "gross_profit", "net_profit"]

    df_sales["price"] = df_sales.family.map(avg_prices)
    df_sales["gross_profit"] = df_sales.sales * df_sales.price

    df_sales["profit_discount_factor"] = promotions_to_profit_discount_factor(
        df_sales.onpromotion, df_sales.price
    )
    df_sales["promotion_cost"] = df_sales.gross_profit * df_sales.profit_discount_factor
    df_sales["net_profit"] = df_sales.gross_profit - df_sales.promotion_cost

    # We don't have a per-product breakdown of transactions, so we'll just sum them out
    # to make the joins to transactions simple
    df_sales = df_sales.groupby(["store_nbr", "date"])[base_columns].sum()

    df_sales = (
        df_sales.join(
            df_transactions.set_index(["store_nbr", "date"]), on=["store_nbr", "date"]
        )
        .fillna(0)
        .reset_index()
    )

    df_sales["report_date"] = df_sales.date + pd.tseries.offsets.MonthEnd(0)
    df_sales = (
        df_sales.groupby(["store_nbr", "report_date"])[base_columns + ["transactions"]]
        .sum()
        .reset_index()
    )

    # Add some additional context to the stores to make the decomposition more
    # interesting
    store_openings = (
        (
            (
                df_sales[df_sales.net_profit != 0]
                .groupby("store_nbr")
                .report_date.first()
                .dt.to_period("M")
                - df_sales.report_date.min().to_period("M")
            ).apply(lambda x: x.n)
        )
        .map(categorise_openings)
        .rename("opening_time_cat")
    )

    df_stores["opening_time_cat"] = df_stores.store_nbr.map(store_openings.to_dict())

    funnel_var = [
        "transactions",
        "items_per_transaction",
        "income_per_item",
        "profit_less_promotions_factor",
    ]

    group_var = [item for item in df_stores.columns if item != "store_nbr"]

    # Convert to funnel analytics
    df_sales["items_per_transaction"] = df_sales.sales / df_sales.transactions
    df_sales["income_per_item"] = df_sales.gross_profit / df_sales.sales
    df_sales["profit_less_promotions_factor"] = (
        df_sales.net_profit / df_sales.gross_profit
    )

    # Truncate data slightly to get a "good read"
    df_sales = df_sales[df_sales.report_date.lt("2017-01-01")][2:]

    return DemoData(
        df_sales=df_sales,
        df_stores=df_stores,
        group_variables=group_var,
        funnel_variables=funnel_var,
    )
