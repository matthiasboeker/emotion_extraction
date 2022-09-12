from pathlib import Path

from spectators.spectator_class import initialise_spectators
from matches.match_report import initialise_match, load_in_match_report
from change_point_detection.clustering import get_goal_break_points
from change_point_detection.regression_test import calculate_f_statistics


if __name__ == "__main__":
    path_to_activity = Path(__file__).parent.parent / "data" / "reduced_files"
    path_to_demographics = Path(
        __file__).parent.parent / "data" / "GENEActiv - soccer match - LIV-MANU - 2022-04-19.csv"
    path_to_match_reports = Path(__file__).parent.parent / "data" / "game_report.csv"
    spectators = initialise_spectators(path_to_activity, path_to_demographics)
    reports = load_in_match_report(path_to_match_reports)
    match = initialise_match("2022-04-19 21:00:00", reports)
    intervals = get_goal_break_points(spectators, match)
    print(calculate_f_statistics(intervals))