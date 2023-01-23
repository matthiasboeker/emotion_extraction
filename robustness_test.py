from pathlib import Path
import pickle
import json

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

from feature_generation.feature_generation import get_feature_df, get_train_test_split
from feature_generation.features import (
    number_peaks,
    ts_std,
    ts_complexity,
    ts_rmsd,
    ts_skew,
    ts_mean,
    ts_kurtosis,
    ts_abs_dev,
    ts_spectral_centroid,
    max_peak,
)
from ml_classification import feature_selection, apply_paired_t_test, apply_tailed_t_test


def load_in_pickle_file(path_to_pickles: Path, file_name: str):
    data = pickle.load(open(path_to_pickles / file_name, "rb"))
    return data


def main():
    path_to_pickles = Path(__file__).parent / "data"
    path_to_figures = Path(__file__).parent / "figures"
    match = load_in_pickle_file(path_to_pickles, "match.pkl")
    spectators = load_in_pickle_file(path_to_pickles, "spectators.pkl")
    goal_times = [goal.real_time for goal in match.goal_events]
    shifts = np.append(np.arange(0, 6000, 100), -np.arange(100, 6000, 100))
    control_times = [{
            "3": 18000 + shift,
            "8": 48000 + shift,
            "27": 162000 + shift,
            "36": 216000 + shift,
        } for shift in shifts]
    res_dummy = []
    res_bound = []
    res_bound_scores = []
    for idx, times in enumerate(control_times):
        feature_df = get_feature_df(
            spectators,
            goal_times,
            times,
            feature_functions={
                   "nr_peaks": number_peaks,
                "max_peak": max_peak,
                "abs_dev": ts_abs_dev,
                "skewness": ts_skew,
                "kurtosis": ts_kurtosis,
                "spectral_centroid": ts_spectral_centroid,
                "std": ts_std,
                "mean": ts_mean,
                "complexity": ts_complexity,
                "rms": ts_rmsd,
            },
            sub_intervals=10
        )
        X_train, y_train = get_train_test_split(
            feature_df,
            test_size=0.0,
            simple_labels=True,
            shuffle=True,
            scaling=True,
            random_state=1,
        )
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        #X_test = X_test.reset_index(drop=True)
        #y_test = y_test.reset_index(drop=True)
        feature_models = {
            "log_reg": LogisticRegression(penalty="l1", solver="liblinear", random_state=0),
            "svc": SVC(C=1, kernel="rbf", random_state=0),
            "ada_boost": AdaBoostClassifier(n_estimators=100, random_state=0),
            "naive_b": GaussianNB(),
            "discriminant": LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
            "knn": KNeighborsClassifier(n_neighbors=10),
            "decision_forest": RandomForestClassifier(random_state=0)

        }
        #selected_features = feature_selection(X_test, y_test)
        #X_train = X_train.loc[:, selected_features]
        #strategy = "prior"
        #clf_dummy = DummyClassifier(strategy=strategy, random_state=0)
        metric_dict = {"accuracy_score": accuracy_score,
                       "matthews_corrcoef": matthews_corrcoef,
                       "roc_auc_score": roc_auc_score
                       }
        for metric, score in {"accuracy_score": 0.50, "matthews_corrcoef": 0.0, "roc_auc_score": 0.5}.items():
            #res_paired_t_test, _ = apply_paired_t_test(
            #        clf_dummy,
            #        feature_models,
            #        X=X_train,
            #        y=y_train,
            #        folds=10,
            #        repetitions=20,
            #        score=metric_dict[metric],
            #    )
            #res_dummy.append({metric: res_paired_t_test})
            res_tailed_test, scores = apply_tailed_t_test(
                    feature_models,
                    X=X_train,
                    y=y_train,
                    folds=10,
                    repetitions=5,
                    score=metric_dict[metric],
                    score_level=score,
                )
            res_bound.append({metric: res_tailed_test})
            res_bound_scores.append({metric: scores})
        print(f"Iteration {idx} of {len(shifts)} is closed")

    #with open('dummy_comp_test.json', 'w') as outfile:
    #    json.dump(res_dummy, outfile)

    with open('lower_bound_test.json', 'w') as outfile:
        json.dump(res_bound, outfile)

    with open('lower_bound_scores.json', 'w') as outfile:
        json.dump(res_bound_scores, outfile)


if __name__ == "__main__":
    main()
