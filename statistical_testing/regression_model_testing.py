from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


def feature_selection(X_train, y_train):
    X_train /= X_train.std(axis=0)
    feature_importance = SelectKBest(mutual_info_classif, k=3).fit(X_train, y_train)
    return feature_importance.get_feature_names_out(X_train.columns)


def calculate_t_statistic(mean_val, variance, folds, repetitions, test_size, train_size):
    return mean_val/np.sqrt((1/(folds * repetitions)+(test_size/train_size))*variance)


def estimate_sample_mean(scores_diff_per_rep, folds, repetitions):
    return np.sum(
        [np.sum(rep_score_diffs) for rep_score_diffs in scores_diff_per_rep]
    ) / (folds * repetitions)


def estimate_sample_variance(scores_diff_per_rep, mean_val, folds, repetitions):
    return np.sum(
        [
            np.sum([(diff - mean_val) ** 2 for diff in cv_diffs])
            for cv_diffs in scores_diff_per_rep
        ]
    ) / (folds * repetitions - 1)


def one_tailed_k_fold_cv_test(
    classifier,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    repetitions: int,
    score: Callable[[..., ...], float],
    score_level
):
    scores_diff_per_rep = []
    pred_cvs_reps = []
    for repetition in range(0, repetitions):
        cross_validation = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=repetition)
        scores_diff_per_cv = []
        ids = X["id"]
        X_ = X.drop(["id"], axis=1)
        pred_cv = []
        for split, (train_index, test_index) in enumerate(cross_validation.split(X_, y, ids)):
            X_cv_train, X_cv_test = X.loc[train_index, :], X.loc[test_index, :]
            y_cv_train, y_cv_test = y.loc[train_index], y.loc[test_index]
            feat = feature_selection(X_cv_train, y_cv_train)
            X_cv_train = X_cv_train.loc[:, feat]
            X_cv_test = X_cv_test.loc[:, feat]
            classifier.fit(X_cv_train, y_cv_train)
            predictions = classifier.predict(X_cv_test)
            pred_cv.append(score(y_cv_test, predictions))
            scores_diff_per_cv.append(
                score(y_cv_test, predictions)-score_level
            )
        scores_diff_per_rep.append(scores_diff_per_cv)
        pred_cvs_reps.append(np.mean(pred_cv))
    #print(f"Mean {score.__name__}: {np.mean(pred_cvs_reps)}")
    mean_val = estimate_sample_mean(scores_diff_per_rep, folds, repetitions)
    variance = estimate_sample_variance(scores_diff_per_rep, mean_val, folds, repetitions)
    t_statistic = calculate_t_statistic(mean_val, variance, folds, repetitions, len(X_cv_test), len(X_cv_train))
    pval = 1 - st.t.cdf(t_statistic, folds * repetitions - 1)
    return pval, pred_cvs_reps


def paired_k_fold_cv_test(
    classifier_a,
    classifier_b,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    repetitions: int,
    score: Callable[[..., ...], float],
):
    scores_diff_per_rep = []
    pred_cvs_reps = []
    for repetition in range(0, repetitions):
        cross_validation = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=repetition)
        ids = X["id"]
        X_ = X.drop(["id"], axis=1)
        scores_diff_per_cv = []
        pred_cv = []
        for split, (train_index, test_index) in enumerate(cross_validation.split(X_, y, ids)):
            X_cv_train, X_cv_test = X.loc[train_index, :], X.loc[test_index, :]
            y_cv_train, y_cv_test = y.loc[train_index], y.loc[test_index]
            feat = feature_selection(X_cv_train, y_cv_train)
            X_cv_train = X_cv_train.loc[:, feat]
            X_cv_test = X_cv_test.loc[:, feat]
            classifier_a.fit(X_cv_train, y_cv_train)
            classifier_b.fit(X_cv_train, y_cv_train)
            predictions_a = classifier_a.predict(X_cv_test)
            predictions_b = classifier_b.predict(X_cv_test)
            pred_cv.append(score(y_cv_test, predictions_b))
            scores_diff_per_cv.append(
                score(y_cv_test, predictions_a) - score(y_cv_test, predictions_b)
            )
        scores_diff_per_rep.append(scores_diff_per_cv)
        pred_cvs_reps.append(np.mean(pred_cv))
    mean_val = estimate_sample_mean(scores_diff_per_rep, folds, repetitions)
    #print(f"Mean {score.__name__}: {np.mean(pred_cvs_reps)}")
    variance = estimate_sample_variance(scores_diff_per_rep, mean_val, folds, repetitions)
    t_statistic = calculate_t_statistic(mean_val, variance, folds, repetitions, len(X_cv_test),len(X_cv_train))
    pval = (1-st.t.cdf(np.abs(t_statistic), folds * repetitions - 1)) * 2
    #pval = st.t.sf(np.abs(t_statistic), len(X)*2 - 1) * 2
    return pval, pred_cvs_reps


class ConfidenceTester:
    def __init__(
        self,
        metric: Callable[[...], float],
        confidence_interval: Dict[str, float],
        t_test_results: Dict[str, float],
    ):
        self.metric = metric
        self.confidence_interval = confidence_interval
        self.t_test_results = t_test_results

    @classmethod
    def test_classification(cls, metric, classifier_results: pd.DataFrame, alpha, mu):
        metrics_per_fold = get_metric_per_fold(classifier_results, metric)
        confidence_interval = calculate_confidence_intervals(metrics_per_fold, alpha)
        t_test_results = t_test(metrics_per_fold, mu)
        return cls(metric, confidence_interval, t_test_results)


def hot_encoding(design_matrix):
    drop_enc = OneHotEncoder(drop="first").fit(design_matrix)
    return drop_enc.transform(design_matrix).toarray()


def prepare_input_data(data):
    y = data["prediction"]
    names = [f"level{i}" for i in range(1, len(set(data["level"])))]
    X = pd.DataFrame(
        hot_encoding(np.array(data["level"]).reshape(-1, 1)), columns=names
    ).iloc[:, 1:]
    return y, X


def calculate_fold_means(data: pd.DataFrame):
    return data.groupby("level").mean().values.tolist()


def get_metric_per_fold(
    cv_results: pd.DataFrame, metric: Callable[[...], float]
) -> List[float]:
    return (
        cv_results.groupby("level")
        .apply(lambda x: metric(x["true"], x["prediction"]))
        .values.tolist()
    )


def calculate_confidence_intervals(fold_metrics: List[float], alpha: float):
    lower_b, upper_b = st.t.interval(
        alpha=alpha,
        df=len(fold_metrics) - 1,
        loc=np.mean(fold_metrics),
        scale=st.sem(fold_metrics),
    )
    return {"lower_bound": lower_b, "upper_bound": upper_b}


def t_test(fold_means: List[float], mu):
    test_statistic, p_value = st.ttest_1samp(
        fold_means, mu, axis=0, alternative="greater"
    )
    return {"test_statistic": test_statistic, "p_value": p_value}


def plot_intervals(results, path_to_save_figure):
    model_names = set([result["model_name"] for result in results])
    metrics = set([result["metric"] for result in results])
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    cm = plt.get_cmap("Set2")
    colors = [cm(1.0 * i / len(model_names)) for i in range(len(model_names))]
    for metric, ax in zip(metrics, axs.ravel()):
        for model_name, c in zip(model_names, colors):
            res = [
                result["res"]
                for result in results
                if (result["model_name"] == model_name) and (result["metric"] == metric)
            ][0]
            intervals = [r.confidence_interval for r in res]
            idx = 0
            for interval in intervals:
                ax.plot(
                    [interval["lower_bound"], interval["upper_bound"]],
                    [idx, idx],
                    "-o",
                    c=c,
                    label=model_name,
                )
                idx += 1
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(), frameon=False, loc="upper left", ncol=1
        )
        ax.set_title(metric.__name__)
        ax.set_ylabel("Splits")
        ax.set_xlabel("Confidence Intervals")
    plt.tight_layout()
    plt.savefig(path_to_save_figure / "Confidence-Intervals.png")


def plot_p_values(results, path_to_save_figure):
    model_names = set([result["model_name"] for result in results])
    metrics = set([result["metric"] for result in results])
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
    cm = plt.get_cmap("Set2")
    colors = [cm(1.0 * i / len(model_names)) for i in range(len(model_names))]
    for metric, ax in zip(metrics, axs.ravel()):
        for model_name, c in zip(model_names, colors):
            res = [
                result["res"]
                for result in results
                if (result["model_name"] == model_name) and (result["metric"] == metric)
            ][0]
            p_values = [r.t_test_results["p_value"] for r in res]
            ax.scatter(
                np.arange(35, 40, 1), p_values, label=model_name, c=c, marker="x"
            )
        ax.legend(frameon=False, loc="upper center", ncol=6)
        ax.set_title(metric.__name__)
        ax.set_ylim(0, 0.2)
        ax.axhline(0.05, c="red")
        ax.set_ylabel("p-values")
        ax.set_xlabel("number of splits")
    plt.tight_layout()
    plt.savefig(path_to_save_figure / "P-Values.png")
