import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# 1. Create the base dataframe
df = pl.DataFrame(
    {
        "A": [1, 1, 1, 1, 1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1, 1, 1, 1, 1],
    }
)

# 2. Add calculated columns
df = df.with_columns([
    (pl.col("A") / pl.col("B")).alias("ratio"),
    (np.log(pl.col("A") / pl.col("B"))).alias("log"),
    (pl.col("A") - pl.col("B")).alias("x_axis")
]).with_columns([
    (pl.col("ratio") - 1).alias("lin_change"),
    (abs(pl.col("ratio") - 1)).alias("abs_lin_change"),
    (pl.col("log") - 0).alias("log_change"),
    (abs(pl.col("log"))).alias("abs_log_change"),
    (pl.col("A") + pl.col("B")).alias("sum")
])

# 3. Add explicit label strings directly to the dataframe
labels = [
    "5 : 1", "4 : 1", "3 : 1", 
    "2 : 1", "1 : 1", "1 : 2", 
    "1 : 3", "1 : 4", "1 : 5"
]
df = df.with_columns(pl.Series("labels", labels))

# 4. Melt the dataframe to 'long format' for Seaborn
df_melted = df.unpivot(
    index=["labels"],
    on=["abs_lin_change", "abs_log_change"],
    variable_name="Scale",
    value_name="Change from 1 m+0 : 1 m+1"
)
# Map the scale names to cleaner text for the legend
df_melted = df_melted.with_columns(
    pl.col("Scale").replace({
        "abs_lin_change": "Linear scale",
        "abs_log_change": "Log scale"
    })
)

# 5. Plotting with Seaborn
sns.set_theme(style="whitegrid") # Apply clean seaborn styling

# Normalize RGBA colors (matplotlib/seaborn require 0-1 range for RGB)
color_red = (114/255, 16/255, 42/255, 1)
color_teal = (2/255, 140/255, 126/255, 1)
custom_palette = [color_red, color_teal]

fig, ax = plt.subplots(figsize=(8, 5))

# Plot lines using seaborn (converting polars to pandas implicitly/explicitly)
sns.lineplot(
    data=df_melted.to_pandas(),
    x="labels",
    y="Change from 1 m+0 : 1 m+1",
    hue="Scale",
    palette=custom_palette,
    marker="o",
    linewidth=2,
    ax=ax
)

ax.legend(title=None)
ax.tick_params(axis="x", rotation=0)
ax.set_xlabel("Ratio m+0 : m+1") 
plt.tight_layout() 
plt.savefig('figures/c7_permutation_invariance.png', dpi=300, bbox_inches='tight', pad_inches=0.12)
plt.show()