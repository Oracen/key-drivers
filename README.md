# Key Driver Analysis

## Introduction
Key Driver Analysis is a statistical technique used to identify the key drivers of a target variable. In the context of KPI analysis within a dimensional model, a key driver is a dimension (or combination of dimensions) that has an outsied influence on the change in target KPI. For example, in a sales dataset, the key drivers of sales might be the store location, the product category, or the type of store.

[//]: <> (Comment: Explain funnel analogy here)

This repository contains some Python helper code to help you perform Key Driver Analysis on your data. It does this through 3 main steps:

1. Metric Decomposition: Decompose the change in the metric into the change in the factors that make up the metric. These factors will be denominated in the value of the top-level metric, and are additive (and thus can be grouped via summation without fear).
2. Category Merging: To clean up many similar-behaving small dimensional categories into a larger, more meaningful category, we apply a clustering procedure to the raw dimensions based on the metric decomposition behaviour. This makes for more meaningful notions of "key drivers".
3. Key Driver Identification: Finally, we apply a process to identify the largest contributors to the change in the target metric. This is done by calculating the change in the target metric for each combination of dimensions, and flagging those with a small membership but large change in the target.

At all stages, the data can be re-sliced, allowing drilldown into specific aspects of data, or summarised into top-level KPIs. The code is designed to be used in conjunction with pandas DataFrames, and can be used within a Jupyter notebook or other Python environment.

### Assumptions
To do this, we make a few assumptions about the data:

1. The target variable is a continuous variable
2. The target variable is a product of the independent variables, and thus can be represented as a funnel
3. The independent variables are generally consistent in sign

In general, the second step can generally be achieved, though you may need to rethink additive relationships (such as losses due to discounts) in order to satisfy this condition.


### Further Reading

Work on general decomposition was motivated by the following posts:

- [Decomposing funnel metrics](https://maxhalford.github.io/blog/funnel-decomposition/)
- [Answering "Why did the KPI change?" using decomposition](https://maxhalford.github.io/blog/kpi-evolution-decomposition/)


## Example Code

Before you can run the code below, you need to install the required packages. If you're using PDM you can do this by running `pdm install` in the root of the repository. Otherwise, you can install the dev dependencies listed in `pyproject.toml`. You'll also need to have a local copy of the `Store Sales - Time Series Forecasting` dataset. You can download this from the [Kaggle page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=stores.csv). If you unpack these files directly into the examples folder, the notebooks should run fine.
