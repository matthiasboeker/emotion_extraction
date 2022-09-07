from pathlib import Path
from typing import List
from dataclasses import dataclass

import pandas as pd
import numpy as np

from matches.match_report import load_in_match_report, initialise_match, Match
from spectators.spectator_class import initialise_spectators, Spectator


@dataclass
class Interval:
    interval_id: str
    interval_series: pd.Series
    team: str


def remove_half_time_indices(activity_ts: pd.DataFrame):
    """TODO: More sophisticated way to extract the times than hard coding """
    first_half = activity_ts.iloc[:46*60,:]
    second_half = activity_ts.iloc[(46+15)*60:, :]
    return pd.concat([first_half, second_half], axis=0).copy().reset_index(drop=True)


def get_supported_team(spec_id, spectators):
    return [spectator.supported_team for spectator in spectators if str(spectator.id) == spec_id][0]


def init_intervals(splits: List[pd.DataFrame], spectators: List[Spectator]):
    intervals = []
    for split in splits:
        for spec_id, activity in split.iteritems():
            intervals.append(Interval(spec_id, activity, get_supported_team(spec_id, spectators)))
    return intervals


def split_df(activity_df, break_points):
    splits = []
    start_index = 0
    for break_point in break_points:
        splits.append(activity_df.iloc[start_index:break_point, :])
        start_index = break_point
    return splits


def get_goal_break_points(spectators: List[Spectator], match: Match):
    activity_full_df = pd.DataFrame(np.array([spectator.activity for spectator in spectators]).T,
                               columns=[str(spectator.id) for spectator in spectators])
    activity_df = remove_half_time_indices(activity_full_df)
    splits = split_df(activity_df, [goal.real_time for goal in match.goal_events])
    return init_intervals(splits, spectators)


if __name__ == "__main__":
    path_to_activity = Path(__file__).parent.parent / "data" / "reduced_files"
    path_to_demographics = Path(
        __file__).parent.parent / "data" / "GENEActiv - soccer match - LIV-MANU - 2022-04-19.csv"
    path_to_match_reports = Path(__file__).parent.parent / "data" / "game_report.csv"
    spectators = initialise_spectators(path_to_activity, path_to_demographics)
    reports = load_in_match_report(path_to_match_reports)
    match = initialise_match("2022-04-19 21:00:00", reports)
    print(get_goal_break_points(spectators, match))