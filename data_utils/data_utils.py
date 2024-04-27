import mat73
import glob
import scipy.io as sio
import numpy as np

from biomarkers import (
    SIOBioMarkers,
    Mat73BioMarkers,
    BioMarkersInterface,
)

AUDIO_BLOCKS = ["audio_hvha", "audio_hvla", "audio_nvha", "audio_nvla"]

def load_data_from_file(file_name: str) -> BioMarkersInterface:
    try:
        raw_data = mat73.loadmat(file_name)
        signal = raw_data["Signal"]
        return Mat73BioMarkers(signal)
    except Exception as ex:
        raw_data = sio.loadmat(file_name)
        signal = raw_data["Signal"]
        return SIOBioMarkers(signal)


def load_data_from_dir(dir_name: str) -> dict:
    print(f"Loading {dir_name} data...")

    # All files and directories ending with .mat and that don't begin with a dot:
    all_files = glob.glob(dir_name + "/*.mat")
    all_data = {}
    for f in all_files:
        markers = load_data_from_file(f)
        block_name = markers.get_block_name()
        all_data[block_name] = markers

    return all

def transform_block_data(block_data):
    # swap trial and time series: (num_channel, num_trials, time_series_per_trial)
    block_data = np.swapaxes(block_data, 1, 2)
    num_trial = block_data.shape[-2]

    d = []
    for t in range(num_trial):
        d.append(block_data[:, t, :])
    # New shape: (num_trials*num_slices_per_trial, num_channel, time_series_per_slice):
    #  (13*num_slices_per_trial, 128, num_seq_per_slice)
    return np.stack(d, axis=0)

def get_block_raw_data_by_marker(subject_data, blocks, marker, channel_no):
    block_to_data = {}
    for b in blocks:
        block_data = subject_data[b]
        transformed_data = transform_block_data(block_data.get_all_data()[marker])
        block_to_data[b] =  transformed_data[:, channel_no, :] 

    return block_to_data