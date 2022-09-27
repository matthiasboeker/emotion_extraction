from typing import Any, Callable, Dict, List
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from feature_generation.feature_generation import get_feature_df
from feature_generation.features import (
    ts_std,
    ts_skew,
    ts_mean,
    ts_rmsd,
    ts_complexity,
    ts_kurtosis,
    ts_abs_dev,
    ts_spectral_centroid,
    max_peak,
    number_peaks,
)


class Times:
    millisecond = 1
    second = millisecond * 100
    minute = second * 60


def load_in_pickle_file(path_to_pickles: Path, file_name: str):
    data = pickle.load(open(path_to_pickles / file_name, "rb"))
    return data


def plot_intervals(goal_intervals, control_intervals, path_to_pictures):
    for goal in range(0, 4):
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        for ticker, ax in zip(goal_intervals, axs.ravel()):
            ax.plot(ticker["intervals"][goal])
            ax.plot(ticker["intervals"][goal].rolling(100).mean())
            ax.set_title(f"Goal {ticker['id']}")
            ax.tick_params(axis="x", labelrotation=45)
        plt.savefig(path_to_pictures / f"activity_plots_goal_{goal}.png")
    for c in range(0, 4):
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        for ticker, ax in zip(control_intervals, axs.ravel()):
            ax.plot(ticker["intervals"][c])
            ax.plot(ticker["intervals"][c].rolling(100).mean())
            ax.set_title(f"Control {ticker['id']}")
            ax.tick_params(axis="x", labelrotation=45)
        plt.savefig(path_to_pictures / f"activity_plots_control_{c}.png")


def pca_features(feature_df: pd.DataFrame):
    pca = PCA(n_components=3)
    pca.fit(feature_df)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    return pca.transform(feature_df)


def map_colors_to_ids(spectators):
    color_list = [
        "red",
        "navy",
        "orange",
        "blue",
        "maroon",
        "lavender",
        "yellowgreen",
        "pink",
        "black",
        "purple",
    ]
    ids = set([spectator.id for spectator in spectators])
    colors = dict(zip(list(ids), color_list))
    return colors


def plot_scatter_fig(data_axis, labels, path_to_pictures, figure_name):
    fig = plt.figure(figsize=(16, 12))
    color = ["orange" if control else "blue" for control in labels["control"]]
    shape = ["d" if control else "x" for control in labels["control"]]
    for x, l, c, s in zip(data_axis[:, 0], data_axis[:, 1], color, shape):
        plt.scatter(x, l, alpha=0.8, c=c, marker=s)
        plt.title("PCA Analysis")
    plt.savefig(path_to_pictures / figure_name)


def main():
    path_to_pickles = Path(__file__).parent / "data"
    path_to_pictures = Path(__file__).parent / "figures"
    match = load_in_pickle_file(path_to_pickles, "match.pkl")
    spectators = load_in_pickle_file(path_to_pickles, "spectators.pkl")
    goal_times = [goal.real_time for goal in match.goal_events]
    # control_times = {"2": 12000, "4": 24000, "8": 48000, "27": 162000, "29": 174000,
    #                 "36": 216000, "49": 390000, "54": 420000}
    control_times = {
        "4": 24000,
        "8": 48000,
        "27": 162000,
        "36": 216000,
        "49": 390000,
        "54": 420000,
    }
    feature_df = get_feature_df(
        spectators,
        goal_times,
        goal_times,
        feature_functions={
            "nr_peaks": number_peaks,
            "max_peak": max_peak,
            "abs_dev": ts_abs_dev,
            "spectral_centroid": ts_spectral_centroid,
            "kurtosis": ts_kurtosis,
            "skew": ts_skew,
            "std": ts_std,
            "mean": ts_mean,
            "complexity": ts_complexity,
            "rms": ts_rmsd,
        },
    )
    labels = feature_df[["control", "id", "goal"]]
    pcs = pca_features(feature_df.drop(["control", "id", "goal"], axis=1))
    plot_scatter_fig(pcs, labels, path_to_pictures, "clustering_scatter_plot_pca.png")


if __name__ == "__main__":
    main()
