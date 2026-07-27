"""
Project : Sales Data Analysis
"""

import pandas as pd

sales = pd.DataFrame({

    "Region":["North","North","South","South","East","West","East","West"],

    "Product":["Laptop","Phone","Laptop","Phone","Laptop","Phone","Phone","Laptop"],

    "Sales":[1200,900,1000,700,1500,800,950,1300]

})

print("="*60)
print("SALES DATA ANALYSIS")
print("="*60)

print("\nDataset")
print(sales)

print("\nAverage Sales by Region")

print(sales.groupby("Region")["Sales"].mean())

print("\nMultiple Aggregations")

print(

sales.groupby("Product").agg(

Average=("Sales","mean"),

Maximum=("Sales","max"),

Minimum=("Sales","min")

)

)

print("\nPivot Table")

pivot = pd.pivot_table(

sales,

values="Sales",

index="Region",

columns="Product",

aggfunc="sum"

)

print(pivot)

print("\nCrossTab")

print(pd.crosstab(sales["Region"],sales["Product"]))
