import numpy as np
import pandas as pd
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Python3Code.util.VisualizeDataset import VisualizeDataset


def add_new_features():
    df = pd.read_csv("./intermediate_datafiles/gestures_v3_clean_enc.csv")
    initial_features = df.columns[1:-4]

    NumAbs = NumericalAbstraction()
    window_sizes = [5, 50, 200]
    for ws in window_sizes:
        df = NumAbs.abstract_numerical(df, initial_features, ws, 'mean')
        df = NumAbs.abstract_numerical(df, initial_features, ws, 'max')
        df = NumAbs.abstract_numerical(df, initial_features, ws, 'min')
        df = NumAbs.abstract_numerical(df, initial_features, ws, 'std')

    print(df.info())

    FreqAbs = FourierTransformation()
    fs = float(1000) / 150
    ws = int(float(10000) / 150)
    df = FreqAbs.abstract_frequency(df, initial_features, ws, fs)

    print(df.info())
    df.to_csv("./intermediate_datafiles/gestures_v3_engi_enc.csv", index_label=False)


def plot_frequencies(feature_name):
    df = pd.read_csv("./intermediate_datafiles/gestures_v3_engi_enc.csv")
    df = df.rename(columns={"time": ""})
    print(df)
    DataViz = VisualizeDataset(__file__)
    frequencies = [col for col in df.columns if col.startswith(feature_name)][:7]
    frequencies.remove(f"{feature_name}_weighted")
    DataViz.plot_dataset(df, [frequencies, "label"], ["exact_like", "like"], ["line", "points"],
                         # legend_ncols=int((len(frequencies)+1) / 4)
                         legend_ncols=3
                         )


plot_frequencies("acc_x_freq")
# add_new_features()
