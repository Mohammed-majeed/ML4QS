import pandas as pd
import pylab
import scipy.stats as stats
import numpy as np

df = pd.read_csv("gestures.csv")
df = df.dropna()


def calculate_statistics(df):
    mean = [df[col].mean() for col in df.columns[1:-1]]
    median = [df[col].median() for col in df.columns[1:-1]]
    max = [df[col].max() for col in df.columns[1:-1]]
    min = [df[col].min() for col in df.columns[1:-1]]
    range = np.subtract(max, min)

    shapiro = [stats.shapiro(df[col])[1] for col in df.columns[1:-1]]
    stats_df = pd.DataFrame({"mean": mean, "median": median, "max": max, "min": min,
                             "range": range, "shapiro": shapiro}, index=df.columns[1:-1])

    return stats_df


stats_df = calculate_statistics(df)
print(stats_df)