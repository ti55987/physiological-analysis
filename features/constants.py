from enum import Enum

class Feature(Enum):
    EEG_SPECTRAL = 28
    DELTA = 1
    THETA = 2
    ALPHA = 3
    BETA1 = 4
    BETA2 = 5
    GAMMA = 6
    ALL_SPECTRAL = 18
    ALL_CHANNEL_SPECTRAL = 19

    ECG_LF = 20
    ECG_HF = 21
    ECG_LFHF = 22

    EGG_FILTERED = 23
    EGG_PHASE = 24
    EGG_AMPLITUDE = 25
    TREV_VELOCITY = 26

    RAW_DATA = 27


EEG_BANDS = {
    Feature.DELTA: (1, 4),
    Feature.THETA: (4, 8),
    Feature.ALPHA: (8, 12),
    Feature.BETA1: (12, 20),
    Feature.BETA2: (20, 30),
    Feature.GAMMA: (30, 50),
}
