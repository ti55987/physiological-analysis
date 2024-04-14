import numpy as np

from sklearn.utils import shuffle

class DatasetBuilder():
    def __init__(self, num_labels: int, val_indexes_group: list):
        self.shuffled_val_indexes = []
        self.shuffled_train_indexes = []
        self.no_shuffled_val_indxes = []
        self.no_shuffled_train_indxes = []

        for val_indexes in val_indexes_group:
            train_indexes = list(set(range(num_labels)) - set(val_indexes))
            self.shuffled_train_indexes.append(shuffle(train_indexes, random_state=0))
            self.shuffled_val_indexes.append(shuffle(val_indexes, random_state=0))
            self.no_shuffled_train_indxes.append(train_indexes)
            self.no_shuffled_val_indxes.append(val_indexes)      

    def set_shuffled_indexes(self, shuffled_train_indexes, shuffled_val_indexes):
        self.shuffled_train_indexes = shuffled_train_indexes
        self.shuffled_val_indexes = shuffled_val_indexes
        
    def get_shuffled_indexes(self):
        return self.shuffled_train_indexes, self.shuffled_val_indexes
    
    def train_test_split(self, data, behavioral_labels, no_shuffle: bool=True):
        dataset = []
        val_indxes = self.no_shuffled_val_indxes if no_shuffle else self.shuffled_val_indexes
        train_label_indxes = self.no_shuffled_train_indxes if no_shuffle else self.shuffled_train_indexes
        
        for idx, val_indexes in enumerate(val_indxes):
            train_labels = np.array(behavioral_labels)[train_label_indxes[idx]]
            val_label = np.array(behavioral_labels)[val_indexes]
            
            train_data_indexes = self.no_shuffled_train_indxes[idx]
            train_data = data[train_data_indexes]
            val_data = data[val_indexes]
            
            dataset.append((train_data, train_labels, val_data, val_label))
        
        return dataset

def get_consecutive_validation_indexes(
    num_train_set: int = 52,
    num_block: int = 4,
    num_slice_per_trial: int = 1,
    start_trial_in_block: int = 1,
    n_step_trial: int = 3.0,
):
    # generate random integer values
    indexes = []
    # extract 3 trials per block
    num_trial_per_block = int(num_train_set / num_block)
    for b in range(0, num_train_set, num_trial_per_block):
        start = b + start_trial_in_block * num_slice_per_trial
        end = start + n_step_trial * num_slice_per_trial
        val_indexes = np.arange(start, end, dtype=int)
        indexes.extend(val_indexes)

    return indexes