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
    mcc_scores = pd.DataFrame([res["matthews_corrcoef"] for res in results if list(res.keys())[0] == "matthews_corrcoef"])
    mcc_scores = mcc_scores.applymap(np.mean)
    auc_scores = pd.DataFrame([res["roc_auc_score"] for res in results if list(res.keys())[0] == "roc_auc_score"])
    auc_scores = auc_scores.applymap(np.mean)
    return {"Accuracy": acc_scores}#{"Accuracy": acc_scores, "MCC": mcc_scores, "Area under ROC": auc_scores}


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


#def plot_boxplots(dfs, save_name):
#    fig, axs = plt.subplots(1, 1)
#    colors = ['lightyellow', 'lightblue', 'lightgreen', "slategrey", "purple"]
#    for ax, metric, df in zip(axs.ravel(), dfs.keys(), dfs.values()):
#        bplot = ax.boxplot(df, patch_artist=True, vert=True , notch=True, labels=["Log. Reg.", "SVC",
#                                                                                  "Naive B",
#                                                                                  "Lin Discriminant",
#                                                                                  "KNN"])
#        #ax.axhline(0.05, c="r")
#        ax.tick_params(axis='x', labelrotation=90)
#        for c, bp in zip(colors, bplot["boxes"]):
#            bp.set_facecolor(c)
#        #ax.set_ylabel("Accuracy")
#        ax.set_title(metric)
#    plt.tight_layout()
#    plt.savefig(Path(__file__).parent/ "figures"/ save_name)


def main():
    path_to_lower_bound_results = Path(__file__).parent / "lower_bound_test.json"
    path_to_dummy = Path(__file__).parent / "dummy_comp_test.json"
    path_to_scores = Path(__file__).parent / "lower_bound_scores.json"
    json_results_lower_b = read_in_experiment_results(path_to_lower_bound_results)
    json_results_dummy = read_in_experiment_results(path_to_dummy)
    json_results_scores = read_in_experiment_results(path_to_scores)
    dfs_lb = regroup_dict_list(json_results_lower_b)
    dfs_cp = regroup_dict_list(json_results_dummy)
    dfs_scores_mean = regroup_dict_list_scores_mean(json_results_scores)
    dfs_scores_var = regroup_dict_list_scores_var(json_results_scores)

    #plot_boxplots(dfs_cp, "robustness_dummy_comp.png")
    #plot_boxplots(dfs_lb, "robustness_lower_bound.png")
    plot_boxplots(dfs_scores_mean, "robustness_lower_bound_scores_new_new_new.png")
    #plot_boxplots(dfs_scores_var, "robustness_lower_bound_scores_var.png")


if __name__ == "__main__":
    main()