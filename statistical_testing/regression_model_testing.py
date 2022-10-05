from typing import List

import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp
from sklearn.preprocessing import OneHotEncoder


def hot_encoding(design_matrix):
    drop_enc = OneHotEncoder(drop='first').fit(design_matrix)
    return drop_enc.transform(design_matrix).toarray()


def prepare_input_data(data):
    y = data["prediction"]
    names = [f"level{i}" for i in range(1, len(set(data["level"])))]
    X = pd.DataFrame(hot_encoding(np.array(data["level"]).reshape(-1, 1)),
                     columns=names).iloc[:, 1:]
    return y, X


def calculate_fold_means(data: pd.DataFrame):
    return data.groupby("level").mean()

def calculate_confidence_intervals(fold_means: List[float]):



def t_test(fold_means: List[float], mu, alternative):
    test_statistic, p_value = ttest_1samp(fold_means, mu, axis=0, alternative=alternative)
    return {"test_statistic": test_statistic, "p_value": p_value}

