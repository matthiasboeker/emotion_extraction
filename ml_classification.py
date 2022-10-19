from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from mlxtend.evaluate import combined_ftest_5x2cv

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
from statistical_testing.regression_model_testing import (
    ConfidenceTester,
    plot_p_values,
    plot_intervals,
    paired_k_fold_cv_test,
    one_tailed_k_fold_cv_test,
)


def pca_dimension_reduction(feature_df: pd.DataFrame, nr_comp):
    pca = PCA(n_components=nr_comp)
    pca.fit(feature_df)
    print(sum(pca.explained_variance_ratio_))
    return pd.DataFrame(pca.transform(feature_df))


def apply_tailed_t_test(feature_models, X, y, folds, repetitions, score, score_level):
    res_t_tests = {}
    for model_name, feature_model in feature_models.items():
        p = one_tailed_k_fold_cv_test(
            feature_model, X, y, folds, repetitions, score, score_level
        )
        res_t_tests[model_name] = p
    return res_t_tests


def apply_paired_t_test(clf_dummy, feature_models, X, y, folds, repetitions, score):
    res_t_tests = {}
    for model_name, feature_model in feature_models.items():
        p = paired_k_fold_cv_test(
            clf_dummy, feature_model, X, y, folds, repetitions, score
        )
        res_t_tests[model_name] = p
    return res_t_tests


def apply_2x5cv(clf_dummy, feature_models, X, y):
    cv5x2_res = {}
    for model_name, feature_model in feature_models.items():
        _, p = combined_ftest_5x2cv(
            estimator1=clf_dummy, estimator2=feature_model, X=X, y=y, random_seed=1
        )
        cv5x2_res[model_name] = p
    return cv5x2_res


def show_model_evaluation(results):
    for model_name, model in results.items():
        res_df = pd.DataFrame(results[model_name])
        print(f"Results for {model_name}:")
        print(f"Mean MCC: {res_df['mcc'].mean()}")
        print(f"Mean AUC: {res_df['auc'].mean()}")
        print(f"Mean ACC: {res_df['acc'].mean()}")
        print("________________________________________________________________")


def evaluate_models(feature_models, cross_validation, X_train, y_train):
    results = {}
    model_prediction_results = {}
    for model_name, model in feature_models.items():
        results[model_name] = []
        statistical_test_data = []
        for split, (train_index, test_index) in enumerate(
            cross_validation.split(X_train, y_train)
        ):
            X_cv_train, X_cv_test = (
                X_train.loc[train_index, :],
                X_train.loc[test_index, :],
            )
            y_cv_train, y_cv_test = y_train.loc[train_index], y_train.loc[test_index]
            model.fit(X_cv_train, y_cv_train)
            y_pred = model.predict(X_cv_test)
            split_array = np.full(y_pred.shape, split, dtype=int)
            statistical_test_data.append(
                np.concatenate(
                    (
                        np.expand_dims(y_pred, 1),
                        np.expand_dims(y_cv_test, 1),
                        np.expand_dims(split_array, 1),
                    ),
                    axis=1,
                )
            )
            mean_acc = model.score(X_cv_test, y_cv_test)
            mcc = matthews_corrcoef(y_cv_test, y_pred)
            auc_score = roc_auc_score(y_cv_test, y_pred)
            cv_results = {"acc": mean_acc, "mcc": mcc, "auc": auc_score}
            results[model_name].append(cv_results)
        model_prediction_results[model_name] = pd.DataFrame(
            np.concatenate(statistical_test_data, axis=0),
            columns=["prediction", "true", "level"],
        )
    return results, model_prediction_results


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


def feature_selection(lasso_path, X_train, y_train, path_to_fig, eps=5e-3):
    X_train /= X_train.std(axis=0)
    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=eps)
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    feature_importance = sort_feature_importance(X_train.columns, coefs_lasso)
    print(feature_importance)
    cm = plt.get_cmap("tab20")
    colors = [cm(1.0 * i / len(X_train.columns)) for i in range(len(X_train.columns))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
    for coef_l, c, name in zip(coefs_lasso, colors, X_train.columns):
        ax1.plot(alphas_lasso, coef_l, c=c, label=name)
        ax2.plot(neg_log_alphas_lasso, coef_l, c=c, label=name)
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("coefficients")
    ax1.set_title("Positive Lasso Alpha")
    ax1.legend(loc="lower right")
    ax2.set_xlabel("-Log(alpha)")
    ax2.set_ylabel("coefficients")
    ax2.set_title("Positive Lasso Neg Log Alpha")
    ax2.legend(loc="lower right")
    plt.savefig(path_to_fig)


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

    X_train, y_train = get_train_test_split(
        feature_df,
        test_size=0,
        simple_labels=True,
        shuffle=True,
        scaling=True,
        random_state=0,
    )
    feature_models = {
        "log_reg": LogisticRegression(penalty="l1", solver="liblinear", random_state=0),
        "svc": SVC(C=1, kernel="rbf", random_state=0),
        "ada_boost": AdaBoostClassifier(n_estimators=50, random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=10),
        "naive_b": GaussianNB(),
        "discriminant": LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
    }
    strategy = "stratified"
    clf_dummy = DummyClassifier(strategy=strategy, random_state=0)
    res_2x5cv = apply_2x5cv(clf_dummy, feature_models, X_train, y_train)
    print(res_2x5cv)
    print(
        f"------------------------- 2X5 CV T-TEST AGAINST {strategy} DUMMY -------------------------"
    )
    res = []
    repetitions = np.arange(1, 21, 1)
    for rep in repetitions:
        res_paired_t_test = apply_paired_t_test(
            clf_dummy,
            feature_models,
            X=X_train,
            y=y_train,
            folds=10,
            repetitions=rep,
            score=accuracy_score,
        )
        res.append(res_paired_t_test)
    res_paired_t_test_df = pd.DataFrame(res)
    print(res_paired_t_test_df)
    print(
        f"------------------------- PAIRED T-TEST AGAINST {strategy} DUMMY -------------------------"
    )
    res = []
    repetitions = np.arange(1, 21, 1)
    for rep in repetitions:
        res_tailed_test = apply_tailed_t_test(
            feature_models,
            X=X_train,
            y=y_train,
            folds=10,
            repetitions=rep,
            score=accuracy_score,
            score_level=0.5,
        )
        res.append(res_tailed_test)
    res_tailed_test_df = pd.DataFrame(res)
    print(res_tailed_test_df)
    print(
        f"------------------------- ONE TAILED PAIRED T-TEST AGAINST -------------------------"
    )


if __name__ == "__main__":
    main()
