import numpy as np
import pandas as pd


def create_dataset(granularity, version, one_hot_encode=False, add_label=False):
    datasets = ["gestures_v3.csv"]
    dfs = [pd.read_csv(f"./datasets/{version}/{dataset}") for dataset in datasets]
    time_offsets = [sum([df["time"].iloc[-1] for df in dfs[:i]]) for i, df in enumerate(dfs)]
    for i, df in enumerate(dfs):
        if add_label:
            label = datasets[i].split(".")[0]
            df["label"] = df.apply(lambda row: "rest" if row["time"] < 10.0 or row["time"] > 1190.0 else label, axis=1)
        df["time"] = pd.to_datetime(df["time"] + time_offsets[i] + 1686920000, unit="s")
        df.drop(columns="Unnamed: 16", axis=1, inplace=True)
    df = pd.concat(dfs).reset_index(drop=True)
    df.rename(columns={"gFx": "gfc_x", "gFy": "gfc_y", "gFz": "gfc_z", "ax": "acc_x", "ay": "acc_y", "az": "acc_z",
                       "wx": "gyr_x", "wy": "gyr_y", "wz": "gyr_z", "Bx": "mag_x", "By": "mag_y", "Bz": "mag_z",
                       "Azimuth": "inc_x", "Pitch": "inc_y", "Roll": "inc_z"}, inplace=True)
    if add_label:
        agg_rules = {col: "mean" for col in df.columns[1:-1]}
        agg_rules["label"] = "first"
    else:
        agg_rules = {col: "mean" for col in df.columns[1:]}

    df = df.resample(f"{granularity}S", on="time").agg(agg_rules).reset_index()
    if one_hot_encode:
        one_hot = pd.get_dummies(df["label"], prefix="label")
        df = df.drop(["label"], axis=1)
        df = df.join(one_hot)
    return df


dataset = create_dataset(granularity=0.15, version="v3")
dataset.to_csv("../intermediate_datafiles/gestures_v3.csv", index=False)

# df = pd.read_csv("../intermediate_datafiles/gestures_v3_clean.csv")
# one_hot = pd.get_dummies(df["label"], prefix="label")
# df = df.drop(["label"], axis=1)
# df = df.join(one_hot)
# df.to_csv("../intermediate_datafiles/gestures_v3_clean_enc.csv", index=False)
