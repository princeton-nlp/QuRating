import argparse
import gzip
import itertools
import json
import numpy as np
import pandas as pd
import pickle
import torch
import tqdm
import time
import faiss
from itertools import chain
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from typing import Dict
from datasets import load_dataset, load_from_disk, concatenate_datasets


def get_top_terms(model_dir, kmeans):
    path_to_vectorizer = model_dir / "tfidf.pkl"
    with open(path_to_vectorizer, 'rb') as f:
        vectorizer = pickle.load(f)
    # this will only work if you use TFIDF vectorizer (which maintains vocab)
    original_space_centroids = vectorizer["svd"].inverse_transform(kmeans.centroids)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    vocab = vectorizer["tfidf"].get_feature_names_out()
    top_terms = []
    for i in range(kmeans.centroids.shape[0]):
        terms = {}
        for j in range(10):
            terms[f"term_{j}"] = vocab[order_centroids[i, j]]
        top_terms.append(terms)
    return pd.DataFrame(top_terms)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def number_normalizer(self, doc, tokenize):
        """Map all numeric tokens to a placeholder.

        For many applications, tokens that begin with a number are not directly
        useful, but the fact that such a token exists can be relevant.  By applying
        this form of dimensionality reduction, some methods may perform better.
        """
        tokens = tokenize(doc)
        return ["#NUMBER" if token[0].isdigit() else token for token in tokens]

    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return (lambda doc: self.number_normalizer(doc, tokenize))


def get_features(input_files, text_field, model_dir, embed_dim):
    ds = concatenate_datasets([load_from_disk(f) for f in input_files])
    # ds = load_dataset("json", data_files=input_files, split="train")

    path_to_vectorizer = model_dir / "tfidf.pkl"
    texts = ds[text_field]

    if not path_to_vectorizer.exists():
        # english stopwords plus the #NUMBER dummy token
        stop_words = list(text.ENGLISH_STOP_WORDS.union(["#NUMBER"]))

        model = Pipeline(
            [("tfidf", NumberNormalizingVectorizer(stop_words=stop_words)),
             ("svd", TruncatedSVD(n_components=embed_dim)),
             ("normalizer", Normalizer(copy=False))])

        vecs = model.fit_transform(tqdm(texts))

        with open(path_to_vectorizer,  "wb+") as f:
            _ = pickle.dump(model, f)
    else:
        with open(path_to_vectorizer, 'rb') as f:
            vectorizer = pickle.load(f)
        vecs = vectorizer.transform(tqdm(texts))
    return vecs

def init_kmeans(args):
    kmeans = faiss.Kmeans(
        args.embed_dim,
        args.num_clusters,
        niter=20,
        verbose=True,
        seed=42,
        gpu=(torch.cuda.device_count() if torch.cuda.is_available() else False),
        spherical=False,
        min_points_per_centroid=1,
        max_points_per_centroid=1000000000,
    )
    if kmeans.cp.spherical:
        kmeans.index = faiss.IndexFlatIP(args.embed_dim)
    else:
        kmeans.index = faiss.IndexFlatL2(args.embed_dim)
    if kmeans.gpu:
        kmeans.index = faiss.index_cpu_to_all_gpus(kmeans.index, ngpu=kmeans.gpu)
    return kmeans

def fit(args):
    if not args.model_dir.is_dir():
        args.model_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    vecs = get_features(args.input_files, args.text_field, args.model_dir, args.embed_dim)
    vecs = vecs[np.random.permutation(vecs.shape[0])]
    print(f"Features computed in {time.time() - start_time} seconds")

    path_to_kmeans = args.model_dir / "kmeans.pkl"

    batches = np.array_split(vecs, np.ceil(vecs.shape[0] / args.batch_size), axis=0)
    kmeans = init_kmeans(args)


    start_time = time.time()
    for i, batch in tqdm(enumerate(batches)):
        print("Batch", i, batch.shape, batch.dtype)
        kmeans.train(batch)

    print(f"Kmeans train took {time.time() - start_time} seconds")

    with open(path_to_kmeans,  "wb+") as f:
        _ = pickle.dump(kmeans.centroids, f)

def predict(args):
    for file in args.input_files:
        vecs = get_features([file], args.text_field, args.model_dir, args.embed_dim)

        kmeans = init_kmeans(args)
        path_to_kmeans = args.model_dir / "kmeans.pkl"

        with open(path_to_kmeans, 'rb') as f:
            kmeans.centroids = pickle.load(f)

        df = get_top_terms(args.model_dir, kmeans)
        df.to_csv(str(args.model_dir / "top_terms.csv"))
        print(df)

        D, I = kmeans.assign(vecs)

        torch.save(torch.from_numpy(I), file + "." + args.model_dir.name + ".assignments")
        torch.save(torch.from_numpy(D), file + "." + args.model_dir.name + ".distances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--input_files", required=True, type=str, nargs="+")
    parser.add_argument("-n", "--num_clusters", required=True, type=int)
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("-f", "--text_field", default="text", type=str)
    parser.add_argument("-m", "--embed_dim", default=128, type=int)
    parser.add_argument("-b", "--batch_size", default=100000, type=int)
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()

    if not args.eval_only:
        fit(args)

    predict(args)