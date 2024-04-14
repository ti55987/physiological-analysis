import numpy as np

from features.constants import EEG_BANDS, Feature

def get_all_spectral_features(feature_to_data: dict):
    all_spetral_psd = [feature_to_data[f.name] for f in EEG_BANDS.keys()]
    all_spetral_psd = np.hstack(all_spetral_psd)
    return all_spetral_psd


def get_all_channel_features_by_bands(data: dict, eeg_bands):
    all_spetral_psd = []
    for _, feature_to_data in data.items():
        spetral_psd = [feature_to_data[f.name] for f in eeg_bands]
        all_spetral_psd.extend(spetral_psd)

    all_spetral_psd = np.hstack(all_spetral_psd)
    return all_spetral_psd


def prepare_eeg_data(data, has_all_spectral, filtered_channel):
    data_dict = {}
    if filtered_channel == "ALL":
        data_dict = {filtered_channel: {}}
        for f in EEG_BANDS.keys():
            data_dict[filtered_channel][f.name] = get_all_channel_features_by_bands(
                data, [f]
            )

        data_dict[filtered_channel][Feature.ALL_SPECTRAL.name] = (
            get_all_channel_features_by_bands(data, list(EEG_BANDS.keys()))
        )

        return data_dict

    data_dict = {}
    for channel, feature_to_data in data.items():
        if channel != filtered_channel and len(filtered_channel) > 0:
            continue

        data_dict[channel] = {}
        if has_all_spectral:
            data_dict[channel][Feature.ALL_SPECTRAL.name] = get_all_spectral_features(
                feature_to_data
            )
            continue

        for f, neural_data in feature_to_data.items():
            # Prepare the data
            data_dict[channel][f] = neural_data

    return data_dict