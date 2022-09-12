from typing import List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pyts.metrics import dtw
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sn


from matches.match_report import load_in_match_report, initialise_match, Match
from spectators.spectator_class import initialise_spectators, Spectator


@dataclass
class Interval:
    interval_id: str
    goal: int
    spectator_id: str
    interval_series: pd.Series
    team: str


def normalise_series(series: pd.Series):
    series_max = min(series)
    series_min = max(series)
    return series.apply(lambda x: (x - series_max) / (series_max - series_min))


def remove_half_time_indices(activity_ts: pd.DataFrame):
    """TODO: More sophisticated way to extract the times than hard coding """
    first_half = activity_ts.iloc[:46*60,:]
    second_half = activity_ts.iloc[(46+15)*60:, :]
    return pd.concat([first_half, second_half], axis=0).copy()


def get_supported_team(spec_id, spectators):
    return [spectator.supported_team for spectator in spectators if str(spectator.id) == spec_id][0]


def init_intervals(splits: List[pd.DataFrame], spectators: List[Spectator]):
    intervals = []
    for index, split in enumerate(splits):
        for spec_id, activity in split.iteritems():
            intervals.append(Interval(f"{index}_{spec_id}", index,str(spec_id), activity, get_supported_team(spec_id, spectators)))
    return intervals


def split_df(activity_df, break_points):
    splits = []
    start_index = 0
    split_indices = break_points + [activity_df.index[-1]]
    for break_point in split_indices:
        splits.append(activity_df.loc[start_index:break_point, :])
        start_index = break_point
    return splits


def get_goal_break_points(spectators: List[Spectator], match: Match):
    activity_full_df = pd.DataFrame(np.array([spectator.activity for spectator in spectators]).T,
                               columns=[str(spectator.id) for spectator in spectators])
    activity_df = remove_half_time_indices(activity_full_df)
    splits = split_df(activity_df, [goal.real_time for goal in match.goal_events])
    return init_intervals(splits, spectators)


def get_dtw_distance_matrix(intervals: List[Interval]):
    series = [normalise_series(interval.interval_series) for interval in intervals]
    return np.asarray([[dtw(p1, p2, method="fast") for p2 in series] for p1 in series])


def most_common(lst):
    return max(set(lst), key=lst.count)


def calculate_purity(result_df: pd.DataFrame):
    most_frequent = []
    for name, group in result_df.groupby("clusters"):
        most_frequent.append(group["goal"].to_list().count(most_common(group["goal"].to_list())))
    return sum(most_frequent)/len(result_df)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig("Dendogram Result")


def plot_distance_matrix(distances_matrix):
    plt.figure(figsize=(10, 7))
    sn.heatmap(distances_matrix)
    plt.savefig("Distance Matrix")


def generate_scatter_plot(results):
    results.groupby("goal").count()
    plt.figure(figsize=(10, 7))
    plt.scatter(results["goal"], results["clusters"], marker="x", c="orange")
    plt.xlabel("Goals")
    plt.ylabel("Clusters")
    plt.savefig("Clustering Result")