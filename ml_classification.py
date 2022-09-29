from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.interval_based import CanonicalIntervalForest

from feature_generation.feature_generation import get_feature_df, get_train_test_split, get_interval_df
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

    #feature_df = get_feature_df(
    #    spectators,
    #    goal_times,
    #    control_times,
    #    feature_functions={
    #        "nr_peaks": number_peaks,
    #        "max_peak": max_peak,
    #        "abs_dev": ts_abs_dev,
    #        #"skewness": ts_skew,
    #        #"kurtosis": ts_kurtosis,
    #        "spectral_centroid": ts_spectral_centroid,
    #        "std": ts_std,
    #        "mean": ts_mean,
    #        "complexity": ts_complexity,
    #        "rms": ts_rmsd,
    #    },
    #)

    intervals_inputs = get_interval_df(spectators,
        goal_times,
        control_times)

    X_train, y_train = get_train_test_split(intervals_inputs, test_size=0, simple_labels=True,
                                                            shuffle=True,
                                                            scaling=False,
                                                            random_state=0)
    #feature_models = {
    #    "log_reg": LogisticRegression(penalty="l1", solver="liblinear",  random_state=0),
    #    "svc": SVC(C=5.0, kernel="rbf",  random_state=0),
    #    "ada_boost": AdaBoostClassifier(n_estimators=50,  random_state=0),
    #    "gaussian_process": GaussianProcessClassifier(kernel=2.0 * RBF(1.0), random_state=0),
    #    "knn": KNeighborsClassifier(n_neighbors=10),
    #    "naive_b": GaussianNB(),
    #    "discriminant": QuadraticDiscriminantAnalysis()
    #}
    ts_models = {
        #"elastic_ensemble": ElasticEnsemble(),
        #"shapelet_classifier": ShapeletTransformClassifier(),
        "rocket_classifier": RocketClassifier(),
        #"canonical_interval_classifier": CanonicalIntervalForest(),

    }
    rkf = RepeatedStratifiedKFold(n_splits=3, n_repeats=20, random_state=0)
    results = {}
    for model_name, model in ts_models.items():
        print(f"Process: {model_name}")
        results[model_name] = []
        for train_index, test_index in rkf.split(X_train, y_train):
            X_cv_train, X_cv_test = X_train.loc[train_index, :], X_train.loc[test_index, :]
            print(X_cv_train.shape)
            y_cv_train, y_cv_test = y_train.loc[train_index], y_train.loc[test_index]
            print("Iteration")
            model.fit(X_cv_train, y_cv_train)
            print("Iteration")
            y_pred = model.predict(X_cv_test)
            mean_acc = model.score(X_cv_test, y_cv_test)
            mcc = matthews_corrcoef(y_cv_test, y_pred)
            auc_score = roc_auc_score(y_cv_test, y_pred)
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








