# Key Drivers
### A Python library for helping making sense of business data

## Introduction
Key Driver Analysis is a statistical technique used to identify the key drivers of a target variable. In the context of KPI analysis within a dimensional model, a key driver is a dimension (or combination of dimensions) that has an outsied influence on the change in target KPI. For example, in a sales dataset, the key drivers of sales might be the store location, the product category, or the type of store.

### Features

This repository contains some Python helper code to help you perform Key Driver Analysis on your data. It does this through 3 main steps:

1. Metric Decomposition: Decompose the change in the metric into the change in the factors that make up the metric. These factors will be denominated in the value of the top-level metric, and are additive (and thus can be grouped via summation without fear).
2. Category Merging: To clean up many similar-behaving small dimensional categories into a larger, more meaningful category, we apply a clustering procedure to the raw dimensions based on the metric decomposition behaviour. This makes for more meaningful notions of "key drivers".
3. Key Driver Identification: Finally, we apply a process to identify the largest contributors to the change in the target metric. This is done by calculating the change in the target metric for each combination of dimensions, and flagging those with a small membership but large change in the target.

At all stages, the data can be re-sliced, allowing drilldown into specific aspects of data, or summarised into top-level KPIs. The code is designed to be used in conjunction with pandas DataFrames, and can be used within a Jupyter notebook or other Python environment.

### Metric Decomposition

We can typically represent any value flow in a company as a series of steps in a funnel; for example, a sales funnel might be represented as:

- Leads become conversions
- Conversions become orders
- Conversions have some average value
- Total of orders gives sales

In many cases, the funnel represents a series of "drop-off" steps, but we can also add in steps beyond conversion to denominate the total value to the organisation. Thinking in this way, as a series of bottlenecks before recording a profit, we can often decompose any value stream into a series of multiplication operation. For example, We can represent the above steps as a formula:

```
sales = leads * conversion_rate * average_value
```

Sometimes, we can also add in additional factors that are not directly related to the funnel, but still contribute to the final value. For example, we might add in a factor for promotions discounting listed sales values, or operating expenses. These factors can be added in as additional terms in the formula, and can be decomposed in the same way (even though the analogy of a funnel is stretching at this point). The key thing is that the target KPI can be expressed as the product of factors, and that these factors are generally consistent in sign. (This condition is mainly for sanity rather than mathematical necessity - even a 0 makes sense so long as not all values are 0.) Similarly, changes in the target KPI can be decomposed into changes in the factors that make up the KPI.

There is one issue with the above example; the units of each of the factors. In the above example, the conversion rate is a percentage, the average value is a currency, and the leads are a count. If we want to go beyond basic trends and make decisions to improve our KPI, we would like to be able to compare the relative importance of each of these factors. To do this, we need to normalise the factors to a common unit. Ideally, we'd like to be able to talk in the unit of the KPI, so that the factors can be compared as a sum rather than a product. This is the purpose of the metric decomposition.

The metric decomposition is a process that takes the change in the target KPI and decomposes it into the change in the factors that make up the KPI. The details are a little arcane, but have a solid geometric argument to support it. Please see the articles listed in the `Further Reading` section, or the attached examples, for more details. The key takeaway is this process can be applied to any KPI that can be represented as a product of factors, and that the factors can be decomposed into a sum in the unit of the target KPI.

### Assumptions
For the metric decomposition to be successful, we make a few assumptions about the data to be treated:

1. The target variable is a continuous variable
2. The target variable is a product of the independent variables, and thus can be represented as a funnel
3. The independent variables are generally consistent in sign
4. The dimensions for analysis are categorical, and can be treated as exchangeable.

In general, the second step can generally be achieved, though you may need to rethink additive relationships (such as losses due to discounts) in order to satisfy this condition.


### Further Reading

Work on general decomposition was motivated by the following posts:

- [Decomposing funnel metrics](https://maxhalford.github.io/blog/funnel-decomposition/)
- [Answering "Why did the KPI change?" using decomposition](https://maxhalford.github.io/blog/kpi-evolution-decomposition/)


## Example Code

Before you can run the code below, you need to install the required packages. If you're using PDM you can do this by running `pdm install` in the root of the repository. Otherwise, you can install the dev dependencies listed in `pyproject.toml`. You'll also need to have a local copy of the `Store Sales - Time Series Forecasting` dataset. You can download this from the [Kaggle page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=stores.csv). If you unpack these files directly into the examples folder, the notebooks should run fine.
