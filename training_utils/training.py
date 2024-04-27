import numpy as np
import cebra

from sklearn.metrics import accuracy_score
from embedding import get_embeddings
from features.labels import get_label_category

MAX_HIDDEN_UNITS = 256

def run_knn_decoder(
    dataset,
    method,
    threshold,
    output_dim,
    max_iterations,
    n_neighbors,
):
    y_pred, y_pred_cat, all_embeddings = [], [], []
    for _, (train_data, train_labels, val_data, _) in enumerate(dataset):
        embedding, val_embedding = get_embeddings(
            train_data=train_data,
            val_data=val_data,
            train_labels=train_labels,
            use_pca=(method == "PCA"),
            out_dim=output_dim,
            num_hidden_units=MAX_HIDDEN_UNITS,
            max_iterations=max_iterations,
        )
        all_embeddings.append(embedding)
        
        # Train the decoder on training embedding and labels
        decoder = cebra.KNNDecoder(n_neighbors=n_neighbors, metric="cosine")
        decoder.fit(embedding, np.array(train_labels))

        prediction = decoder.predict(val_embedding)
        y_pred.append(prediction)
        y_pred_cat.append([0 if p < threshold else 1 for p in prediction])

    return y_pred, y_pred_cat, all_embeddings

def decode_marker_data(
    dataset_dict,
    label_type,
    v_thred,
    a_thred,
    method,
    output_dim,
    threshold,
    max_iteration,
):
    accuracy = {k: {} for k in dataset_dict.keys()}
    list_embedding_tuple = []
    for channel, feature_to_data in dataset_dict.items():
        if len(feature_to_data) == 0:
            continue

        for f, dataset in feature_to_data.items():
            val_true_cat = [
                get_label_category(val_labels, label_type, v_thred, a_thred)
                for _, (_, _, _, val_labels) in enumerate(dataset)
            ]
            
            _, val_pred_cat, all_embeddings = run_knn_decoder(
                dataset,
                method,
                threshold,
                output_dim,
                max_iteration,
                16,
            )

            ac_scores = [
                accuracy_score(y_pred=val_pred_cat[i], y_true=val_true_cat[i])
                for i in range(len(val_pred_cat))
            ]

            max_score_index = np.array(ac_scores).argmax(axis=0)
            mean_acc = round(np.mean(ac_scores),2) 
            max_acc = round(ac_scores[max_score_index], 2)
            list_embedding_tuple.append(
                (
                    f"CV Acc Max: {max_acc} Avg:{mean_acc}",
                    max_score_index,
                    all_embeddings[max_score_index],
                    (dataset[max_score_index][1], dataset[max_score_index][3]),
                )
            )

            accuracy[channel][f] = ac_scores
       
    return list_embedding_tuple, accuracy

def get_metadata(accuracy: dict):
    all_channels, all_feature_name, cv_scores = [], [], []
    for k, v in accuracy.items():
        for f, score in v.items():
            cv_scores.append(score)
            all_channels.append(k)
            all_feature_name.append(f)

    return all_channels, all_feature_name, cv_scores