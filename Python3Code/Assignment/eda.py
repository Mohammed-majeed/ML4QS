import pandas as pd
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def calculate_statistics(df):
    mean = [df[col].mean() for col in df.columns[1:-1]]
    median = [df[col].median() for col in df.columns[1:-1]]
    max = [df[col].max() for col in df.columns[1:-1]]
    min = [df[col].min() for col in df.columns[1:-1]]
    range = np.subtract(max, min)
    shapiro = [stats.shapiro(df[col]).pvalue for col in df.columns[1:-1]]
    stats_df = pd.DataFrame({"mean": mean, "median": median, "max": max, "min": min,
                             "range": range, "shapiro": shapiro}, index=df.columns[1:-1])
    print(stats_df)


def plot_feature_over_time(df, feature):
    length = len(df)
    x = range(length)
    y = df[feature][:length]
    color = ["red" if df["label"].iloc[i] == "rest"
             else "blue" if df["label"].iloc[i] == "wave"
             else "green" if df["label"].iloc[i] == "handshake"
             else "orange" if df["label"].iloc[i] == "clap"
             else "purple" for i in range(len(y))]
    plt.scatter(x, y, color=color, s=2)
    plt.show()


df = pd.read_csv("../intermediate_datafiles/gestures_clean.csv")
df = df.dropna()

df_wave = df.loc[df["label"] == "wave"][:50]

fig, axs = plt.subplots(3)
axs[0].plot(pd.to_datetime(df_wave["time"]), df_wave["acc_x"])
axs[1].plot(pd.to_datetime(df_wave["time"]), df_wave["acc_y"])
axs[2].plot(pd.to_datetime(df_wave["time"]), df_wave["acc_z"])
plt.show()

# plot_feature_over_time(df, "gfc_x")
