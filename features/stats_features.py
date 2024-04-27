import numpy as np
from scipy import stats
from enum import Enum

AXIS = 0

class StatsFeature(Enum):
    # statistical features
    MEAN = 1
    STD = 2
    PTP = 3
    VAR = 4
    MINIM = 5
    MAXIM = 6
    MEAN_SQUARE = 7
    RMS = 8
    ABS_DIFF = 9
    SKEWNESS = 10
    KURTOSIS = 11


# mean is the average of the data
def mean(data):
    return np.mean(data, axis=AXIS)

# std is the standard deviation
def std(data):
    return np.std(data, axis=AXIS)

# ptp indicates peak to peak
def ptp(data):
    return np.ptp(data, axis=AXIS)

# var is the variance of the data
def var(data):
    return np.var(data, axis=AXIS)

# The minimum of the data
def minim(data):
    return np.min(data, axis=AXIS)

# The maximum of the data
def maxim(data):
    return np.max(data, axis=AXIS)

def mean_square(data):
    return np.mean(data ** 2, axis=AXIS)

# root mean square.
def rms(data):
    return np.sqrt(np.mean(data ** 2, axis=AXIS))

def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data, axis=AXIS)), axis=AXIS)

# skewness is a measure of the asymmetry of the probability distribution
# of a real-valued random variable about its mean.
def skewness(data):
    return stats.skew(data, axis=AXIS)

# kurtosis is a measure of the "tailedness" of the probability distribution
# of a real-valued random variable.
def kurtosis(data):
    return stats.kurtosis(data, axis=AXIS)


FEATURE_TO_FUNC = {
    StatsFeature.MEAN: mean,
    StatsFeature.STD: std,
    StatsFeature.PTP: ptp,
    StatsFeature.VAR: var,
    StatsFeature.MINIM: minim,
    StatsFeature.MAXIM: maxim,
    StatsFeature.MEAN_SQUARE: mean_square,
    StatsFeature.RMS: rms,
    StatsFeature.ABS_DIFF: abs_diffs_signal,
    StatsFeature.SKEWNESS: skewness,
    StatsFeature.KURTOSIS: kurtosis,
}