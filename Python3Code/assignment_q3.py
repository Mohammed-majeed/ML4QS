import os

import numpy
import pandas as pd
import matplotlib.pyplot as plt

from Python3Code.Chapter3.OutlierDetection import DistanceBasedOutlierDetection, DistributionBasedOutlierDetection
# from Python3Code.Chapter3.KalmanFilters import KalmanFilters
from Python3Code.Chapter3.ImputationMissingValues import ImputationMissingValues

df = pd.read_csv("./intermediate_datafiles/gestures_v3.csv")

outliers = DistanceBasedOutlierDetection()

final_df = pd.DataFrame()
label_lof_dict = {"wave": 1.3, "handshake": 1.3, "clap": 1.3, "rest": 1.05}
for label, lof_limit in label_lof_dict.items():
    df_label = df.loc[df["label"] == label]
    df_label = outliers.local_outlier_factor(df_label, df.columns[1:-1], "euclidean", k=5)
    print(f"Number of outliers: {len(df_label.loc[df_label['lof'] > lof_limit])}")
    print(f"Percentage of outliers: {len(df_label.loc[df_label['lof'] > lof_limit]) / len(df_label['lof'])}")
    colors = ["red" if lof > lof_limit else "blue" for i, lof in enumerate(df_label["lof"])]

    # plt.scatter(pd.to_datetime(df_label["time"]), df_label["lof"], c=colors, s=0.3)
    # plt.show()

    df_label.loc[df_label["lof"] > lof_limit, df_label.columns[1:-2]] = numpy.NaN
    final_df = pd.concat([final_df, df_label])

df = final_df.drop(["lof"], axis=1).sort_index().reset_index()
miss_val = ImputationMissingValues()
for col in df.columns[1:]:
    df = miss_val.impute_interpolate(df, col)

df.to_csv("./intermediate_datafiles/gestures_v3_clean.csv", index=False)
print(df)
print(df.info())
