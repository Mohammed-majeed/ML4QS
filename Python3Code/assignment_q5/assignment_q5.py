import operator

import pandas as pd
import numpy as np

filename = "gestures_fea_eng_no_ohe"
df = pd.read_csv(f"../intermediate_datafiles/{filename}.csv")
train = df.drop(["time", "label"], axis=1)


def get_num_unique_values():
    num_unique = {}
    for col in train:
        num_unique[col] = train[col].nunique()

    num_unique = dict(sorted(num_unique.items(), key=operator.itemgetter(1)))
    ratio_num_unique = sum([1 if val > (max(num_unique.values()) * 0.95) else 0 for val in num_unique.values()]) / len(
        num_unique)
    print(num_unique)
    print(ratio_num_unique)


def get_feature_importance():
    feature_importance = pd.read_csv("../intermediate_datafiles/feature_importance_v2.csv")
    random_forest = feature_importance[["Feature", "Random Forest Importance"]]
    random_forest = random_forest.sort_values(by=["Random Forest Importance"], ascending=False)
    print(random_forest)

    for num_features in [10, 50, 100]:
        features = random_forest["Feature"][:num_features].tolist()
        selected_features = df[features + ["label"]]
        selected_features.to_csv(f"../intermediate_datafiles/feature_selection/gestures_v3_engi_top{num_features}.csv",
                                 index=False)


def train_test_split_gestures(val_split=0.):
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_top10 = pd.read_csv("../intermediate_datafiles/feature_selection/gestures_feat_engi_top10.csv").fillna(0)

    for label in ["wave", "clap", "handshake", "high five"]:
        label_df = df_top10.loc[df["label"] == label]
        label_train_df, label_val_df, label_test_df = np.split(label_df, [int(0.5 * len(label_df)),
                                                                          int(0.7 * len(label_df))])
        train_df = pd.concat([train_df, label_train_df], ignore_index=True)
        if val_split == 0:
            train_df = pd.concat([train_df, label_val_df], ignore_index=True)
        val_df = pd.concat([val_df, label_val_df], ignore_index=True)
        test_df = pd.concat([test_df, label_test_df], ignore_index=True)

    rest = df_top10.loc[df["label"] == "rest"]
    rest = [group for _, group in rest.groupby(rest.index.to_series().diff().ne(1).cumsum())][-1]
    rest_train_df, rest_val_df, rest_test_df = np.split(rest, [int(0.5 * len(rest)), int(0.7 * len(rest))])
    train_df = pd.concat([train_df, rest_train_df], ignore_index=True)
    if val_split == 0:
        train_df = pd.concat([train_df, rest_val_df], ignore_index=True)
    val_df = pd.concat([val_df, rest_val_df], ignore_index=True)
    test_df = pd.concat([test_df, rest_test_df], ignore_index=True)
    if val_split == 0:
        return train_df, test_df
    return train_df, val_df, test_df


def train_test_split_gestures_v3(val_split=0.):
    df = pd.read_csv("../intermediate_datafiles/feature_selection/gestures_v3_engi_top10.csv").fillna(0)
    df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].min()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min())

    train_df, val_df, test_df = np.split(df, [int(0.5 * len(df)),
                                              int(0.7 * len(df))])
    if val_split == 0.:
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        return train_df, test_df
    return train_df, val_df, test_df


train_test_split_gestures_v3()
