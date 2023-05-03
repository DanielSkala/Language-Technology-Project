import pickle

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def visualize_embeddings(embedding_objects):
    # Get embeddings
    embeddings = list(embedding_objects.values())

    # Reduce dimensionality to 3D
    pca = PCA(n_components=3)
    pca.fit(embeddings)
    embeddings_3D = pca.transform(embeddings)

    # Get names
    names = list(embedding_objects.keys())

    # Create dataframe
    df = pd.DataFrame(embeddings_3D, columns=["x", "y", "z"])
    df["category"] = names

    # Plot
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="category", hover_name="category")
    fig.update_layout(showlegend=False)
    fig.show()


if __name__ == "__main__":
    with open("value_categories_embeddings.pkl", "rb") as f:
        embedded_value_categories = pickle.load(f)
        visualize_embeddings(embedded_value_categories)

    with open("arguments_embeddings.pkl", "rb") as f:
        embedded_arguments = pickle.load(f)
        visualize_embeddings(embedded_arguments)
