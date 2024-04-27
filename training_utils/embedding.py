
from sklearn.decomposition import PCA
import cebra
from cebra import CEBRA

def model_fit(
    neural_data,
    out_dim,
    num_hidden_units,
    behavioral_labels,
    max_iterations: int = 10000,
):
    single_cebra_model = CEBRA(
        # model_architecture = "offset10-model",
        batch_size=512,
        temperature_mode="auto",
        #device='cpu',
        output_dimension=out_dim,
        max_iterations=max_iterations,
        num_hidden_units=num_hidden_units,
        verbose = False,
    )

    if behavioral_labels is None:
        single_cebra_model.fit(neural_data)
    else:
        single_cebra_model.fit(neural_data, behavioral_labels)
    
    #cebra.plot_loss(single_cebra_model)
    return single_cebra_model


def tsne_visualization(components):
    from sklearn.manifold import TSNE
    tsne2d = TSNE(n_components=2, random_state=0) 
    return tsne2d.fit_transform(components) 


def get_embeddings(
    train_data,
    val_data,
    train_labels,
    use_pca: bool = False,
    out_dim: int = 8,
    num_hidden_units: int = 256,
    max_iterations: int = 100,
):
    if use_pca:
        # Run PCA
        pca = PCA(n_components=out_dim)
        pca = pca.fit(train_data)
        return pca.transform(train_data), pca.transform(val_data)

    single_cebra_model = model_fit(train_data, out_dim, num_hidden_units, train_labels, max_iterations)

    # Calculate embedding
    embedding = single_cebra_model.transform(train_data)
    val_embedding = single_cebra_model.transform(val_data)
    return embedding, val_embedding