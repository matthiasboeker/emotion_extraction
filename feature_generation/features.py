import numpy as np
from numpy.fft import rfft
from scipy.stats import kurtosis, skew, entropy, median_abs_deviation
from scipy.signal import find_peaks


def ts_mean(activity_interval):
    return np.mean(activity_interval)


def ts_std(activity_interval):
    return np.std(activity_interval)


def ts_skew(activity_interval):
    return skew(activity_interval.diff()[1:])


def ts_kurtosis(activity_interval):
    return kurtosis(activity_interval.diff()[1:])


def ts_entropy(activity_interval):
    return entropy(activity_interval)


def ts_abs_dev(activity_interval):
    return median_abs_deviation(activity_interval)


def ts_count_zero(activity_interval):
    return activity_interval.loc[activity_interval == 0].sum()


def number_peaks(activity_interval):
    return len(find_peaks(activity_interval)[0])


def ts_complexity(activity_interval):
    return np.sqrt(activity_interval.diff()[1:].apply(lambda x: x ** 2).mean())


def ts_rmsd(activity_interval):
    return np.sqrt(activity_interval.apply(lambda x: x ** 2).mean())


def max_peak(activity_interval):
    return max(activity_interval)


def ts_spectral_centroid(signal):
    spectrum = abs(np.fft.rfft(signal))
    normalized_spectrum = spectrum / sum(spectrum)  # like a probability mass function
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    return sum(normalized_frequencies * normalized_spectrum)
