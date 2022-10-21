from pathlib import Path
import pickle
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


from feature_generation.feature_generation import get_feature_df, get_train_test_split
from feature_generation.features import (
    number_peaks,
    max_peak,
    ts_std,
    ts_mean,
    ts_rmsd,
    ts_skew,
    ts_kurtosis,
    ts_complexity,
    ts_abs_dev,
    ts_spectral_centroid,
)


def extract_interval_nr(feature_name: str):
    return int(re.findall(r"\d+", feature_name)[0])  # feature_name[-1]


def extract_feature_name(feature_name: str):
    return re.sub(r"[0-9]", "", feature_name)  # feature_name[:-1]


def transform_feature_importance(feature_importance: list[tuple[str, int]]):
    return pd.DataFrame(
        {
            "feature": [
                extract_feature_name(importance_tuple[0])
                for importance_tuple in feature_importance
                if importance_tuple[1] > 0
            ],
            "interval": [
                extract_interval_nr(importance_tuple[0])
                for importance_tuple in feature_importance
                if importance_tuple[1] > 0
            ],
            "score": [
                importance_tuple[1]
                for importance_tuple in feature_importance
                if importance_tuple[1] > 0
            ],
        }
    )


def sort_feature_importance(feature_names, feature_coefficients):
    return sorted(
        {
            feature_name: len(feature_coefficients[0])
            - np.count_nonzero(feature_coeffs == 0)
            for feature_name, feature_coeffs in zip(feature_names, feature_coefficients)
        }.items(),
        key=lambda x: x[1],
        reverse=True,
    )


def get_group(dataframe, id):
    return dataframe.loc[dataframe["id"] == id]


def get_overall_feature_importance(lasso_path, X_total, eps=5e-3):
    ids = list(set(X_total["id"]))
    res = []
    for id in ids:
        X_spec = get_group(X_total, id)
        y_spec = X_spec["control"]
        X_spec = X_spec.drop(["id", "control", "goal"], axis=1)
        feature_importance_spec = transform_feature_importance(
            feature_selection(lasso_path, X_spec, y_spec, eps=eps)
        )
        feature_importance_spec["id"] = np.array(
            [id for _ in range(0, len(feature_importance_spec))]
        )
        res.append(feature_importance_spec)
    return res, pd.concat(res, axis=0)


def forest_feature_evaluation_(X_total):
    ids = list(set(X_total["id"]))
    res = []
    for id in ids:
        X_spec = get_group(X_total, id)
        y_spec = X_spec["control"]
        X_spec = X_spec.drop(["id", "control", "goal"], axis=1)
        feature_importance_spec = tree_based_feature_selection(X_spec, y_spec, id)
        res.append(feature_importance_spec)
    fig, axs = plt.subplots(5, 2, figsize=(20, 15))
    colors = [
        "darkgreen",
        "orange",
        "maroon",
        "navy",
        "grey",
        "black",
        "purple",
        "goldenrod",
        "lightgreen",
        "olive",
    ]
    for ax, df, c in zip(axs.ravel(), res, colors):
        ax.bar(df["features"], df["mean_importance"], yerr=df["std_importance"], color=c)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel("Features")
        ax.set_ylabel("Feature Importance Score")
        ax.set_title(f"Goal Interval Detection of {df['id'].iat[0]}")
    plt.tight_layout()
    plt.show()


def forest_feature_evaluation(X_total):
    y_spec = X_total["control"]
    X_spec = X_total.drop(["id", "control", "goal"], axis=1)
    feature_importance_spec = tree_based_feature_selection(X_spec, y_spec, 1)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.bar(feature_importance_spec["features"], feature_importance_spec["mean_importance"], yerr=feature_importance_spec["std_importance"])
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel("Features")
    ax.set_ylabel("Feature Importance Score")
    ax.set_title(f"Goal Interval Detection")
    plt.tight_layout()
    plt.show()


def tree_based_feature_selection(X, y, id):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    forest = RandomForestClassifier(random_state=0, max_depth=10)
    forest.fit(X_train, y_train)
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importance = pd.Series(result.importances_mean, index=forest.feature_names_in_)
    return pd.DataFrame({"mean_importance": forest_importance,
                         "std_importance": std,
                         "features": forest.feature_names_in_,
                         "id": [id for _ in range(0, len(std))]})



def feature_selection(lasso_path, X_train, y_train, eps=5e-3):
    X_train /= X_train.std(axis=0)
    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=eps)
    feature_importance = sort_feature_importance(X_train.columns, coefs_lasso)
    return feature_importance


def scatter_plot_feature_importance(feature_imp_list, path_to_fig):
    fig, axs = plt.subplots(5, 2, figsize=(15, 10))
    markers = [".", ",", "o", "<", ">", "s", "p", "*", "+", "h"]
    colors = [
        "darkgreen",
        "orange",
        "maroon",
        "navy",
        "grey",
        "black",
        "purple",
        "goldenrod",
        "lightgreen",
        "olive",
    ]
    features_colors = {
        feature: color
        for feature, color in zip(
            list(set(pd.concat(feature_imp_list, axis=0)["feature"])), colors
        )
    }

    for ax, df, m in zip(axs.ravel(), feature_imp_list, markers):
        df = df.sort_values(["interval"], ascending=[True])
        cols = [features_colors[feature] for feature in df["feature"]]
        ax.scatter(df["interval"], df["score"], marker=m, c=cols)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel("Interval")
        ax.set_ylabel("Feature Importance Score")
        ax.set_title(f"Goal Interval Detection of {df['id'].iat[0]}")
    plt.tight_layout()
    plt.show()
    plt.savefig(path_to_fig / "feature_importance_scatter_plot.png")


def load_in_pickle_file(path_to_pickles: Path, file_name: str):
    data = pickle.load(open(path_to_pickles / file_name, "rb"))
    return data


def main():
    path_to_pickles = Path(__file__).parent / "data"
    path_to_figures = Path(__file__).parent / "figures"
    match = load_in_pickle_file(path_to_pickles, "match.pkl")
    spectators = load_in_pickle_file(path_to_pickles, "spectators.pkl")
    goal_times = [goal.real_time for goal in match.goal_events]
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
        control_times,
        feature_functions={
            #"nr_peaks": number_peaks,
             "max_peak": max_peak,
             #"abs_dev": ts_abs_dev,
             #"skewness": ts_skew,
             #"kurtosis": ts_kurtosis,
             #"spectral_centroid": ts_spectral_centroid,
             #"std": ts_std,
             #"mean": ts_mean,
             #"complexity": ts_complexity,
             #"rms": ts_rmsd,
        },
        sub_intervals=60,
    )

    X_train, y_train = get_train_test_split(
        feature_df,
        test_size=0,
        simple_labels=True,
        shuffle=True,
        scaling=True,
        random_state=10,
    )

    #detection_list, detection_df = get_overall_feature_importance(
    #    lasso_path, feature_df, eps=0.01
    #)
    #scatter_plot_feature_importance(detection_list, path_to_figures)
    forest_feature_evaluation(feature_df)

if __name__ == "__main__":
    main()
