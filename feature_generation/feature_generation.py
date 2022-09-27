from pathlib import Path
from typing import Any, Callable
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from spectators.spectator_class import Spectator


class Times:
    millisecond = 1
    second = millisecond * 100
    minute = second * 60


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


def merge_feature_lists(feature_lists: dict[str, list]):
    return pd.DataFrame(feature_lists["control"] + feature_lists["goal"])


def get_feature_df(spectators, goal_times, control_times, feature_functions):
    goal_intervals = get_intervals_for_spectators(spectators, goal_times, 1)
    control_intervals = get_intervals_for_spectators(
        spectators, control_times.values(), 1
    )
    control_features = get_ts_features_from_intervals(
        control_intervals, feature_functions, control=True
    )
    goal_features = get_ts_features_from_intervals(
        goal_intervals, feature_functions, control=False
    )
    return merge_feature_lists(
        {"control": control_features, "goal": goal_features}
    ).fillna(method="ffill")


def get_train_test_split(feature_df, test_size):
    labels = feature_df[["id", "goal", "control"]]
    feature_df.drop(["control", "id", "goal"], axis=1)
    return train_test_split(feature_df, labels, test_size=test_size)
