from pathlib import Path
from typing import Any, Callable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from spectators.spectator_class import Spectator


class Times:
    millisecond = 1
    second = millisecond * 100
    minute = second * 60


def plot_intervals(goal_intervals, control_intervals, length_goal, length_control, path_to_pictures):
    for goal in range(0, length_goal):
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        for ticker, ax in zip(goal_intervals, axs.ravel()):
            ax.plot(ticker["intervals"][goal])
            ax.plot(ticker["intervals"][goal].rolling(100).mean())
            ax.set_title(f"Goal {ticker['id']}")
            ax.tick_params(axis="x", labelrotation=45)
        plt.savefig(path_to_pictures / f"activity_plots_goal_{goal}.png")
    for c in range(0, length_control):
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        for ticker, ax in zip(control_intervals, axs.ravel()):
            ax.plot(ticker["intervals"][c])
            ax.plot(ticker["intervals"][c].rolling(100).mean())
            ax.set_title(f"Control {ticker['id']}")
            ax.tick_params(axis="x", labelrotation=45)
        plt.savefig(path_to_pictures / f"activity_plots_control_{c}.png")


def get_goal_interval(spectator: Spectator, goal_times: list[int], extra_range):
    spectator_intervals = {}
    spectator_intervals["id"] = spectator.id
    spectator_intervals["intervals"] = []
    for goal_time in goal_times:
        spectator_intervals["intervals"].append(
            spectator.activity.loc[
                goal_time - extra_range : goal_time + Times.minute + extra_range
            ]
        )
    return spectator_intervals


def get_intervals_for_spectators(
    spectators: list[Spectator], goal_times: list[int], extra_range
):
    res = []
    for spectator in spectators:
        res.append(get_goal_interval(spectator, goal_times, extra_range))
    return res


def feature_extraction(
    activity_interval: pd.Series, feature_functions: dict[str, Callable[..., float]]
):
    feature_dict = {}
    for feature_name, feature_function in feature_functions.items():
        feature_dict[feature_name] = feature_function(activity_interval)
    return feature_dict


def get_ts_features_from_intervals(
    spectators: list[dict[str, Any]],
    feature_functions: dict[str, Callable[..., float]],
    control: bool,
) -> list[dict[str, float]]:
    res = []
    for spectator in spectators:
        for goal, interval in enumerate(spectator["intervals"]):
            feature = feature_extraction(interval, feature_functions)
            feature["id"] = spectator["id"]
            feature["control"] = control
            feature["goal"] = goal
            res.append(feature)
    return res


def get_raw_intervals(
    spectators: list[dict[str, Any]],
    control: bool,
) -> list[dict[str, float]]:
    res = []
    for spectator in spectators:
        for goal, interval in enumerate(spectator["intervals"]):
            feature = {"interval": interval}
            feature["id"] = spectator["id"]
            feature["control"] = control
            feature["goal"] = goal
            res.append(feature)
    return res


def merge_lists(feature_lists: dict[str, list]):
    return feature_lists["control"] + feature_lists["goal"]


def get_interval_df(spectators, goal_times, control_times):
    goal_intervals = get_intervals_for_spectators(spectators, goal_times, 1)
    control_intervals = get_intervals_for_spectators(
        spectators, control_times.values(), 1
    )
    control_features = get_raw_intervals(
        control_intervals, control=True
    )
    goal_features = get_raw_intervals(
        goal_intervals, control=False
    )
    return pd.DataFrame(merge_lists(
        {"control": control_features, "goal": goal_features}
    ))


def get_feature_df(spectators, goal_times, control_times, feature_functions):
    goal_intervals = get_intervals_for_spectators(spectators, goal_times, 1)
    control_intervals = get_intervals_for_spectators(
        spectators, control_times.values(), 1
    )
    #plot_intervals(goal_intervals, control_intervals, len(goal_times), len(control_times),
    #               Path(__file__).parent.parent/ "figures")
    control_features = get_ts_features_from_intervals(
        control_intervals, feature_functions, control=True
    )
    goal_features = get_ts_features_from_intervals(
        goal_intervals, feature_functions, control=False
    )
    return pd.DataFrame(merge_lists(
        {"control": control_features, "goal": goal_features}
    )).fillna(method="ffill")


def get_train_test_split(feature_df: pd.DataFrame, test_size: float, simple_labels: bool,
                         scaling: bool, **kwargs):
    labels = feature_df[["id", "goal", "control"]]
    if simple_labels:
        labels = feature_df["control"]
    output_df = feature_df.drop(["control", "id", "goal"], axis=1)
    feature_names = output_df.columns
    if scaling:
        scaler = StandardScaler()
        output_df = pd.DataFrame(scaler.fit_transform(output_df), columns=feature_names)
    if test_size == 0:
        return output_df, labels
    return train_test_split(output_df, labels, test_size=test_size, **kwargs)
