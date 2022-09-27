from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from feature_generation.feature_generation import get_feature_df, get_train_test_split
from feature_generation.features import number_peaks, max_peak, ts_std, ts_mean, ts_rmsd, ts_skew, ts_kurtosis, \
    ts_complexity, ts_abs_dev, ts_spectral_centroid


def load_in_pickle_file(path_to_pickles: Path, file_name: str):
    data = pickle.load(open(path_to_pickles / file_name, "rb"))
    return data


def main():
    path_to_pickles = Path(__file__).parent / "data"
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
            "nr_peaks": number_peaks,
            "max_peak": max_peak,
            "abs_dev": ts_abs_dev,
            "spectral_centroid": ts_spectral_centroid,
            "std": ts_std,
            "mean": ts_mean,
            "complexity": ts_complexity,
            "rms": ts_rmsd,
        },
    )
    X_train, X_test, y_train, y_test = get_train_test_split(feature_df, test_size=0.1, simple_labels=True,
                                                            shuffle=False, random_state=0)
    models = {
        "log_reg": LogisticRegression(penalty="l1", solver="liblinear"),
        "svc": SVC(C=1.0, kernel="rbf"),
        "ada_boost": AdaBoostClassifier(n_estimators=50)
    }
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    results = {}
    for model_name, model in models.items():
        results[model_name] = []
        for train_index, test_index in rkf.split(X_train, y_train):
            X_cv_train, X_cv_test = X_train.loc[train_index, :], X_train.loc[test_index, :]
            y_cv_train, y_cv_test = y_train.loc[train_index], y_train.loc[test_index]
            model.fit(X_cv_train, y_cv_train)
            y_pred = model.predict(X_cv_test)
            mean_acc = model.score(X_cv_test, y_cv_test)
            mcc = matthews_corrcoef(y_cv_test, y_pred)
            auc_score = auc(y_cv_test, y_pred)
            cv_results = {"acc": mean_acc, "mcc": mcc, "auc": auc_score}
            results[model_name].append(cv_results)

        res_df = pd.DataFrame(results[model_name])
        print(f"Results for {model_name}:")
        print(f"Mean MCC: {res_df['mcc'].mean()}")
        print(f"Mean AUC: {res_df['auc'].mean()}")
        print(f"Mean ACC: {res_df['acc'].mean()}")
        print("________________________________________________________________")


if __name__ == "__main__":
    main()








