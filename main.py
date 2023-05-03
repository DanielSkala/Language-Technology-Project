from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

import pickle
from tqdm import tqdm
import csv
import json

from value_categories import VALUE_CATEGORIES

MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def embed_arguments(arguments):
    embeddings = {}

    for argument_id, argument_text in tqdm(arguments.items()):
        embedding = MODEL.encode(argument_text, convert_to_tensor=True)
        embeddings[argument_id] = embedding.cpu().numpy()

    return embeddings


def embed_value_categories(value_categories):
    embeddings = {}

    for category_name, category_info in tqdm(value_categories.items()):
        description = category_info["description"]
        text = f"{category_name}: {description}"
        embedding = MODEL.encode(text, convert_to_tensor=True)
        embeddings[category_name] = embedding.cpu().numpy()

    return embeddings


def classify_argument(argument: str, value_embeddings):
    # Embed the input argument
    argument_embedding = MODEL.encode(argument, convert_to_tensor=True).cpu().numpy()

    # Calculate cosine similarity between argument and value category embeddings
    similarities = {}
    for category_name, category_embedding in value_embeddings.items():
        similarity = cosine_similarity([argument_embedding], [category_embedding])
        similarities[category_name] = similarity[0][0]

    # turn the similarities into a softmax distribution
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

    # get the softmax distribution
    softmax_similarities = softmax(list(similarities.values()))

    # print sorted softmax similarities
    for category_name, similarity in sorted(zip(similarities.keys(), softmax_similarities),
                                            key=lambda x: x[1], reverse=True):
        print(f"{category_name}: {similarity}")

    # print sorted softmax similarities
    for category_name, similarity in sorted(zip(similarities.keys(), softmax_similarities),
                                            key=lambda x: x[1], reverse=True):
        print(f"{category_name}: {similarity}")

    # get the best matching category
    best_matching_category = list(similarities.keys())[np.argmax(softmax_similarities)]

    return best_matching_category


if __name__ == "__main__":
    embedded_value_categories = embed_value_categories(VALUE_CATEGORIES)
    with open("value_categories_embeddings.pkl", "wb") as f:
        pickle.dump(embedded_value_categories, f)

    # with open("value_categories_embeddings.pkl", "rb") as f:
    #     embedded_value_categories = pickle.load(f)

    # Embed arguments
    with open("./datasets/arguments-training.tsv", "r") as file:
        reader = csv.DictReader(file, delimiter="\t")
        arguments = {row["Conclusion"]: row["Premise"] for row in reader}
    embedded_arguments = embed_arguments(arguments)
    with open("arguments_embeddings.pkl", "wb") as f:
        pickle.dump(embedded_arguments, f)

    # with open("arguments_embeddings.pkl", "rb") as f:
    #     embedded_arguments = pickle.load(f)

    # Use first argument as example
    # with open("./datasets/arguments-training-small.tsv", "r") as file:
    #     reader = csv.DictReader(file, delimiter="\t")
    #     for row in tqdm(reader):
    #         category = classify_argument(row["Premise"], embedded_value_categories)
    #         print(f"Argument ID: {row['Conclusion']}\nValue Category: {category}\n")
