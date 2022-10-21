from pathlib import Path
from typing import Any, Callable, Union
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


def get_interval(spectator: Spectator, times: list[int], extra_range):
    spectator_intervals = {}
    spectator_intervals["id"] = spectator.id
    spectator_intervals["intervals"] = []
    for time in times:
        spectator_intervals["intervals"].append(
            spectator.activity.loc[
                time - extra_range : time + Times.minute + extra_range
            ]
        )
    return spectator_intervals


def get_intervals_for_spectators(
    spectators: list[Spectator], goal_times: list[int], extra_range
):
    res = []
    for spectator in spectators:
        res.append(get_interval(spectator, goal_times, extra_range))
    return res


def feature_extraction(
    activity_interval: pd.Series, feature_functions: dict[str, Callable[..., float]], sub_intervals: Union[None, int]
) -> dict[str, float]:
    feature_dict = {}
    for feature_name, feature_function in feature_functions.items():
        if sub_intervals != None:
            feature_dict[f"{feature_name}{sub_intervals}"] = feature_function(activity_interval)
        else:
            print(sub_intervals)
            feature_dict[feature_name] = feature_function(activity_interval)
    return feature_dict


def get_ts_features_from_intervals(
    spectators: list[dict[str, Any]],
    feature_functions: dict[str, Callable[..., float]],
    sub_intervals: Union[None, int],
    control: bool,
) -> list[dict[str, float]]:
    res = []
    for spectator in spectators:
        for goal, interval in enumerate(spectator["intervals"]):
            if sub_intervals:
                sub_length = int(len(interval)/sub_intervals)
                beg = 0
                feature = {}
                for idx, chunk in enumerate(range(sub_length, len(interval), sub_length)):
                    sub_feature = feature_extraction(interval.iloc[beg:chunk], feature_functions, idx)
                    beg = chunk
                    feature.update(sub_feature)
            else:
                feature = feature_extraction(interval, feature_functions, None)
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


def get_feature_df(spectators, goal_times, control_times, feature_functions, sub_intervals: Union[None, int]):
    goal_intervals = get_intervals_for_spectators(spectators, goal_times, 1)
    control_intervals = get_intervals_for_spectators(
        spectators, control_times.values(), 1
    )
    control_features = get_ts_features_from_intervals(
        control_intervals, feature_functions, sub_intervals, control=True)
    goal_features = get_ts_features_from_intervals(
        goal_intervals, feature_functions, sub_intervals, control=False)
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
