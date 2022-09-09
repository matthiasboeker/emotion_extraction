from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from spectators.spectator_class import initialise_spectators
from matches.match_report import initialise_match, load_in_match_report
from change_point_detection.clustering import calculate_purity, get_dtw_distance_matrix, get_goal_break_points


def plot_results(results):

    fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=3, ncols=1, figsize=(15,10))
    ax_1.plot([res["purity"] for res in results["complete"]], linestyle='--', marker='o', color='b', label="purity")
    ax_1.plot([res["rand"] for res in results["complete"]], linestyle='--', marker='o', color='orange', label="rand")
    ax_1.set_title("complete")
    ax_1.legend()
    ax_2.plot([res["purity"] for res in results["average"]], linestyle='--', marker='o', color='b',label="purity")
    ax_2.plot([res["rand"] for res in results["average"]], linestyle='--', marker='o', color='orange',label="rand")
    ax_2.set_title("average")
    ax_2.legend()
    ax_3.plot([res["purity"] for res in results["single"]], linestyle='--', marker='o', color='b',label="purity")
    ax_3.plot([res["rand"] for res in results["single"]],linestyle='--', marker='o', color='orange', label="rand")
    ax_3.set_title("single")
    ax_3.legend()
    plt.savefig("Results.png")

def run_experiment(params, intervals, distances_matrix):
    results = {}
    for linkage in params["linkage"]:
        results[linkage] = []
        for distance in params["distance"]:
            results[linkage].append(evaluate_clustering(intervals, linkage, distance, distances_matrix))
    return results


def evaluate_clustering(intervals, linkage, distance, distances_matrix):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, compute_full_tree=True,
                                         affinity="precomputed", linkage=linkage).fit(distances_matrix)
    results = pd.DataFrame({
        "clusters": clustering.labels_,
        "goal": [interval.goal for interval in intervals]
    })
    return {"purity": calculate_purity(results),
            "rand": adjusted_rand_score(results["goal"], results["clusters"])}


if __name__ == "__main__":
    path_to_activity = Path(__file__).parent.parent / "data" / "reduced_files"
    path_to_demographics = Path(
        __file__).parent.parent / "data" / "GENEActiv - soccer match - LIV-MANU - 2022-04-19.csv"
    path_to_match_reports = Path(__file__).parent.parent / "data" / "game_report.csv"
    spectators = initialise_spectators(path_to_activity, path_to_demographics)
    reports = load_in_match_report(path_to_match_reports)
    match = initialise_match("2022-04-19 21:00:00", reports)
    intervals = get_goal_break_points(spectators, match)
    distances_matrix = get_dtw_distance_matrix(intervals)
    params = {"linkage": ["complete", "average", "single"],
              "distance": np.arange(5, 20, 0.5)}
    results = run_experiment(params, intervals, distances_matrix)
    plot_results(results)
    with open('cluster_results.json', 'w') as outfile:
        json.dump(results, outfile)
