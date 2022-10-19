from typing import Callable, Dict, Tuple
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import signal


def extract_match_interval(dataframe: pd.DataFrame, match_times: Dict[str, str]):
    start_index = dataframe.time.index[dataframe.time == match_times["start"]][0]
    end_index = dataframe.time.index[dataframe.time == match_times["end"]][0]
    return dataframe.iloc[start_index:end_index, :]


def pca_combination(match_dataframe: pd.DataFrame) -> np.array:
    vectors = match_dataframe[["acc_x", "acc_y", "acc_z"]]
    pca = PCA(n_components=1, svd_solver="full")
    return pca.fit_transform(vectors)[:, 0]


def vector_combination(match_dataframe: pd.DataFrame) -> np.array:
    vectors = match_dataframe[["acc_x", "acc_y", "acc_z"]].apply(
        lambda x: np.sqrt(x["acc_x"]**2 + x["acc_y"]**2 + x["acc_z"]**2), axis=1)
    return vectors


def butter_bandpass(lowcut, highcut, fs, order=5) -> Tuple[np.array, np.array]:
    return signal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='ba')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def filter_signal(
    signal_ts: np.array, bin_array: np.array
) -> np.array:
    filtered_signal = abs(butter_bandpass_filter(signal_ts, 1, 40, 100))
    return filtered_signal #np.digitize(abs(filtered_signal), bins=bin_array)


def acceleration_to_count(
    match_df: pd.DataFrame,
    combination_func: Callable[[pd.DataFrame], np.array],
    bin_array=np.arange(0, 5, 0.0390625),
):
    combined_signal = combination_func(match_df)
    filtered_signal = filter_signal(combined_signal, bin_array)
    return (
        pd.DataFrame(
            {
                "time": match_df.time.apply(lambda x: x[:22]).reset_index(drop=True),
                "signal": filtered_signal,
            }
        )
        .groupby("time")
        .max()
    )


def read_in_activity_file(path_to_file):
    columns = [
        "time",
        "acc_x",
        "acc_y",
        "acc_z",
        "lux",
        "event marker",
        "linear active thermistor",
    ]
    return pd.read_csv(path_to_file, skiprows=100, header=None, names=columns)


def store_reduced_files(
    path_to_folder, path_to_output, match_times, vector_combination_func
):
    for index_nr, file_name in enumerate(
        [
            file_name
            for file_name in os.listdir(path_to_folder)
            if file_name.startswith("__")
        ]
    ):
        dataframe = read_in_activity_file(path_to_folder / file_name)
        match_dataframe = extract_match_interval(dataframe, match_times)
        activity_count_signal = acceleration_to_count(
            match_dataframe,
            vector_combination_func,
        )
        activity_count_signal.to_csv(path_to_output / file_name)
        print(f"Stored File {file_name}")


if __name__ == "__main__":
    path_to_data = Path(__file__).parent.parent / "data"
    path_to_output = path_to_data / "reduced_files"
    match_times = {"start": "2022-04-19 21:00:00:000", "end": "2022-04-19 22:49:00:000"}
    store_reduced_files(
        path_to_data, path_to_output, match_times, vector_combination
    )
