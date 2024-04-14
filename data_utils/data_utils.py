import mat73
import glob
import scipy.io as sio

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

    return all_data
