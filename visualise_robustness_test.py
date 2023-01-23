from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_in_experiment_results(path_to_results: Path):
    with open(path_to_results, 'rb') as f:
        return json.load(f)


def regroup_dict_list(results):
    acc_scores = pd.DataFrame([res["accuracy_score"] for res in results if list(res.keys())[0] == "accuracy_score"])
    mcc_scores = pd.DataFrame([res["matthews_corrcoef"] for res in results if list(res.keys())[0] == "matthews_corrcoef"])
    auc_scores = pd.DataFrame([res["roc_auc_score"] for res in results if list(res.keys())[0] == "roc_auc_score"])
    return {"Accuracy": acc_scores, "MCC": mcc_scores, "Area under ROC": auc_scores}


def regroup_dict_list_scores_mean(results):
    acc_scores = pd.DataFrame([res["accuracy_score"] for res in results if list(res.keys())[0] == "accuracy_score"])
    acc_scores = acc_scores.applymap(np.mean)
    return {"Accuracy": acc_scores}


def regroup_dict_list_scores_var(results):
    acc_scores = pd.DataFrame([res["accuracy_score"] for res in results if list(res.keys())[0] == "accuracy_score"])
    acc_scores = acc_scores.applymap(np.var)
    mcc_scores = pd.DataFrame([res["matthews_corrcoef"] for res in results if list(res.keys())[0] == "matthews_corrcoef"])
    mcc_scores = mcc_scores.applymap(np.var)
    auc_scores = pd.DataFrame([res["roc_auc_score"] for res in results if list(res.keys())[0] == "roc_auc_score"])
    auc_scores = auc_scores.applymap(np.var)
    return {"Accuracy": acc_scores, "MCC": mcc_scores, "Area under ROC": auc_scores}


def plot_boxplots(dfs, save_name):
    fig, ax = plt.subplots(1, 1)
    colors = ['lightyellow', 'lightblue', 'lightgreen', "slategrey", "purple", "blue", "olive"]
    bplot = ax.boxplot(dfs["Accuracy"], whis = [5, 95], patch_artist=True, vert=True, notch=True, labels=["Log. Reg.", "SVC",
                                                                                  "AdaBoost",
                                                                                  "Naive B",
                                                                                  "Lin Discriminant",
                                                                                  "k-NN",
                                                                                "Random Forest"])
    ax.tick_params(axis='x', labelrotation=90)
    for c, bp in zip(colors, bplot["boxes"]):
        bp.set_facecolor(c)
    ax.axhline(0.5, c="r")
    ax.axhline(0.6, c="r")
    ax.axhline(0.549, c="r")
    ax.set_title("Accuracy")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent/ "figures"/ save_name)


def main():
    path_to_scores = Path(__file__).parent / "lower_bound_scores.json"
    json_results_scores = read_in_experiment_results(path_to_scores)
    dfs_scores_mean = regroup_dict_list_scores_mean(json_results_scores)
    plot_boxplots(dfs_scores_mean, "robustness_lower_bound_scores.png")


if __name__ == "__main__":
    main()