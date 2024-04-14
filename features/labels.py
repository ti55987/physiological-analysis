
def get_user_rating_raw_labels(subject_data, blocks):
    v_labels, a_labels, attention_labels = [], [], []
    for b in blocks:
        block_data = subject_data[b]
        v_labels.extend(block_data.get_labels().flatten())
        a_labels.extend(block_data.get_labels('arousal').flatten())
        attention_labels.extend(block_data.get_labels('attention').flatten())

    return v_labels, a_labels, attention_labels

def get_label_category(labels, label_type, v_thred, a_thred):
    threshold = a_thred if label_type == "arousal" else v_thred
    return [0 if p < threshold else 1 for p in labels]