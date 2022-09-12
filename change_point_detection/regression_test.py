from typing import List
from itertools import groupby

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import f

from change_point_detection.clustering import Interval



def visual_eval_regression(regression_data, results):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(15, 10))
    ax1.scatter(regression_data["x_t"], regression_data["x_t-1"], marker="x", c="black", label="1")
    ax1.legend()
    ax2.scatter(regression_data["x_t"], regression_data["x_t-2"], marker=".", c="green", label="2")
    ax2.legend()
    ax3.scatter(regression_data["x_t"], np.log(regression_data["x_t-3"]), marker="<", c="orange", label="3")
    ax3.legend()
    plt.show()
    fig = sm.qqplot(results.resid, fit=True)
    plt.show()


def fit_regression_model(interval: pd.Series):
    regression_data = pd.DataFrame({"x_t" : interval[3:],
                                    "x_t-1": interval.shift(1)[3:],
                                    "x_t-2": interval.shift(2)[3:],
                                    "x_t-3": interval.shift(3)[3:]})
    design_matrix = regression_data[["x_t-1", "x_t-2", "x_t-3"]]
    design_matrix = sm.add_constant(design_matrix)
    model = sm.OLS(regression_data["x_t"], design_matrix)
    results = model.fit()
    return results


def calculate_f_statistic(combined_results, result_part_one, result_part_two):
    number_model_params = len(combined_results.params)+len(result_part_one.params)+len(result_part_two.params)
    f_statistic_numerator = (combined_results.ssr - (result_part_one.ssr + result_part_two.ssr))/number_model_params
    f_statistic_denominator = (result_part_one.ssr + result_part_two.ssr)/\
                              (len(result_part_one.params)+len(result_part_two.params)+2*number_model_params)
    return 1-f.cdf(f_statistic_numerator/f_statistic_denominator,
                   len(combined_results.params), len(result_part_one.params)+len(result_part_two.params))


def calculate_f_statistics(intervals: List[Interval]):
    sorted_intervals = sorted(intervals, key=lambda x: x.spectator_id)
    test_results = {}
    for spec_id, group in groupby(sorted_intervals, key=lambda interval: interval.spectator_id):
        spectators_intervals = sorted(list(group), key=lambda interval: interval.goal)
        f_stats = []
        for i in range(0, len(spectators_intervals)-1):
            common_results = fit_regression_model(pd.concat([spectators_intervals[i].interval_series,
                                         spectators_intervals[i+1].interval_series]))
            part_one_results = fit_regression_model(spectators_intervals[i].interval_series)
            part_two_results = fit_regression_model(spectators_intervals[i+1].interval_series)
            f_stats.append(calculate_f_statistic(common_results, part_one_results, part_two_results))
        test_results[spec_id] = f_stats
    return test_results

