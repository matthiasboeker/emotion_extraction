from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from feature_generation.feature_generation import get_feature_df, get_train_test_split
from feature_generation.features import number_peaks, max_peak, ts_std, ts_mean, ts_rmsd, ts_skew, ts_kurtosis, \
    ts_complexity, ts_abs_dev, ts_spectral_centroid


def evaluate_various_models(feature_models, cross_validation, X_train, y_train):
    results = {}
    for model_name, model in feature_models.items():
        print(f"Process: {model_name}")
        results[model_name] = []
        for train_index, test_index in cross_validation.split(X_train, y_train):
            X_cv_train, X_cv_test = X_train.loc[train_index, :], X_train.loc[test_index, :]
            y_cv_train, y_cv_test = y_train.loc[train_index], y_train.loc[test_index]
            model.fit(X_cv_train, y_cv_train)
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


def sort_feature_importance(feature_names, feature_coefficients):
    return sorted({feature_name:  len(feature_coefficients[0])-np.count_nonzero(feature_coeffs == 0) for feature_name,
                                                                                                feature_coeffs in
            zip(feature_names, feature_coefficients)}.items(), key= lambda x: x[1], reverse=True)


def feature_selection(lasso_path, X_train, y_train, path_to_fig, eps=5e-3):
    X_train /= X_train.std(axis=0)
    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=eps)
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    feature_importance = sort_feature_importance(X_train.columns, coefs_lasso)
    print(feature_importance)

    cm = plt.get_cmap('tab20')
    colors = [cm(1. * i / len(X_train.columns)) for i in range(len(X_train.columns))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
    for coef_l, c, name in zip(coefs_lasso, colors, X_train.columns):
        ax1.plot(alphas_lasso, coef_l, c=c, label=name)
        ax2.plot(neg_log_alphas_lasso, coef_l, c=c, label=name)
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("coefficients")
    ax1.set_title("Positive Lasso Alpha")
    ax1.legend(loc='lower right')
    ax2.set_xlabel("-Log(alpha)")
    ax2.set_ylabel("coefficients")
    ax2.set_title("Positive Lasso Neg Log Alpha")
    ax2.legend(loc='lower right')
    plt.savefig(path_to_fig)


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
            "skewness": ts_skew,
            "kurtosis": ts_kurtosis,
            "spectral_centroid": ts_spectral_centroid,
            "std": ts_std,
            "mean": ts_mean,
            "complexity": ts_complexity,
            "rms": ts_rmsd,
        },
    )

    X_train, y_train = get_train_test_split(feature_df, test_size=0, simple_labels=True,
                                                            shuffle=True,
                                                            scaling=True,
                                                            random_state=0)
    feature_models = {
        "log_reg": LogisticRegression(penalty="l1", solver="liblinear",  random_state=0),
        "svc": SVC(C=2.0, kernel="rbf",  random_state=0),
        "ada_boost": AdaBoostClassifier(n_estimators=50,  random_state=0),
        "gaussian_process": GaussianProcessClassifier(kernel=2.0 * RBF(1.0), random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=10),
        "naive_b": GaussianNB(),
        "discriminant": LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    }

    path_to_figure = Path(__file__).parent / "figures" / "l1-penalty-evaluation.png"
    #feature_selection(lasso_path, X_train, y_train, path_to_figure)
    rkf = RepeatedStratifiedKFold(n_splits=3, n_repeats=20, random_state=0)
    evaluate_various_models(feature_models, rkf, X_train, y_train)

if __name__ == "__main__":
    main()








