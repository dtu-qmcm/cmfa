import numpy as np
import polars as pl

import matplotlib.pyplot as plt

df = pl.DataFrame(
    {
        "A": [1, 1, 1, 1, 1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1, 1, 1, 1, 1],
    }
)

df.insert_column(column=(pl.col("A")/pl.col("B")).alias("ratio"), index=2)
df.insert_column(column=(pl.col("ratio")-1).alias("lin_change"), index=3)
df.insert_column(column=(abs(pl.col("ratio")-1)).alias("abs_lin_change"), index=4)
df.insert_column(column=(np.log(pl.col("A")/pl.col("B"))).alias("log"), index=5)
df.insert_column(column=(pl.col("log")-0).alias("log_change"), index=6)
df.insert_column(column=(abs(pl.col("log"))).alias("abs_log_change"), index=7)
df.insert_column(column=(pl.col("A")-pl.col("B")).alias("x_axis"), index=8)
df.insert_column(column=(pl.col("A")+pl.col("B")).alias("sum"),index=9)

fig, ax = plt.subplots(1,1)
ax.plot(df["x_axis"].to_numpy(), df["abs_lin_change"].to_numpy(), label='Linear scale')
ax.plot(df["x_axis"].to_numpy(), df["abs_log_change"].to_numpy(), label='Log scale')
ax.set_xticks(df["x_axis"], labels=("5 m0 : 1 m1","4 m0 : 1 m1", "3 m0 : 1 m1","2 m0 : 1 m1","1 m0 : 1 m1", "1 m0 : 2 m1", "1 m0 : 3 m1", "1 m0 : 4 m1","1 m0 : 5 m1"))
ax.tick_params(axis="x", length=3, labelrotation= 55)
ax.set_ylabel("Change from 1 m0 : 1 m1")
ax.legend()
plt.show()