from pathlib import Path
import pickle
from spectators.spectator_class import initialise_spectators, Spectator
from matches.match_report import load_in_match_report, initialise_match
from ssa_factorisation.factorise_ssa_matrices import SSA


def pickle_file(path_to_pickle, file_name, serialisable_obj):
    pickle.dump(serialisable_obj, open(path_to_pickle /file_name, "wb"))


if __name__ == "__main__":
    path_to_activity = Path(__file__).parent / "data" / "reduced_files"
    path_to_demographics = (
        Path(__file__).parent
        / "data"
        / "GENEActiv - soccer match - LIV-MANU - 2022-04-19.csv"
    )
    path_to_match_reports = Path(__file__).parent / "data" / "game_report.csv"

    spectators = initialise_spectators(path_to_activity, path_to_demographics)
    reports = load_in_match_report(path_to_match_reports)
    match = initialise_match("2022-04-19 21:00:00", reports)
    ssa_objs = {spectator.id: SSA.transform_fit(spectator.activity, window_size=60*15, lag=60*5, q=5)
            for spectator in spectators}

    path_to_save_files = Path(__file__).parent / "data"
    pickle_file(path_to_save_files, "spectators.pkl", spectators)
    pickle_file(path_to_save_files, "reports.pkl", reports)
    pickle_file(path_to_save_files, "match.pkl", match)
    pickle_file(path_to_save_files, "ssa_objs.pkl", ssa_objs)