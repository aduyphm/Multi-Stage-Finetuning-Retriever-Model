#!/usr/bin/env python
# coding: utf-8

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import pickle
import os

def word_segment(sent):
    sent = tokenize(sent.encode('utf-8').decode('utf-8'))
    return sent

# Hyper parameters
k1 = 1.2
b = 0.75

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--train_file", default="dvc.json", type=str)
    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--corpus_file", default="legal_truncated_corpus.json", type=str)
    parser.add_argument("--top_k", default=20, type=int)
    args = parser.parse_args()

    data_dir = os.path.join(os.getcwd(), args.data_dir)
    data_file = args.data_file
    train_file = args.train_file
    corpus_file = args.corpus_file
    top_k = args.top_k

    if not os.path.exists(os.path.join(data_dir, corpus_file)):
        with open(os.path.join(data_dir, data_file), 'r') as json_file:
            queries_data = json.load(json_file)
        train = []
        corpus = []
        idx = 0
        for item in queries_data["data"]:
            title = item["title"]
            for paragraph in item["paragraphs"]:
                for qa in paragraph["qas"]:
                    train.append({
                        "question": qa["question"],
                        "related_docs": [str(idx)]
                    })
                corpus.append({
                    "id": str(idx),
                    "title": title,
                    "context": paragraph["context"]
                })
                idx += 1
        with open(os.path.join(data_dir, corpus_file), 'w') as json_file:
            json.dump(corpus, json_file)
    else:
        with open(os.path.join(data_dir, train_file), 'r') as json_file:
            train = json.load(json_file)

    # Read corpus
    if corpus_file.endswith(".json"):
        corpus_df = pd.read_json(os.path.join(data_dir, corpus_file))
    elif corpus_file.endswith(".csv"):
        corpus_df = pd.read_csv(os.path.join(data_dir, corpus_file), encoding="utf-16")

    if not os.path.join(data_dir, f"bm25_{corpus_file.split('.')[0]}"):
        bm25_corpus = []
        for index, row in corpus_df.iterrows():
            p = word_segment(row["context"])
            p_tokens = p.split()
            bm25_corpus.append(p_tokens)

        bm25 = BM25Plus(bm25_corpus, k1=k1, b=b)
        with open(os.path.join(data_dir, f"bm25_{corpus_file.split('.')[0]}"), "wb") as bm_file:
            pickle.dump(bm25, bm_file)
        
    # Load model
    with open(os.path.join(data_dir, f"bm25_{corpus_file.split('.')[0]}"), "rb") as bm_file:
        bm25 = pickle.load(bm_file)
        
    data = []
    for i, item in tqdm(enumerate(train)):
        query = item["question"]
        query = query[:-1] if query.endswith("?") else query
        relevant_docs = item["related_docs"]
        actual_positive = len(relevant_docs)

        query_seg = word_segment(query)
        query_tokens = query_seg.split()
        doc_scores = bm25.get_scores(query_tokens)

        predictions = np.argpartition(doc_scores, len(doc_scores) - top_k)[-top_k:]
        hits1 = corpus_df.iloc[predictions]
        hits = hits1.copy()
        hits["id"] = hits["id"].astype(str)
        hits["score"] = doc_scores[:top_k] / (np.linalg.norm(doc_scores[:top_k])+1e-20)

        hits_sorted = hits.sort_values(by=['id', 'score'], ascending=[True, False])
        result = hits_sorted.groupby(['id']).first()

        neg_docs = []
        neg_scores = []
        for j, idx_pred in enumerate(predictions):
            key = hits.at[idx_pred, "id"]
            if key not in relevant_docs:
                neg_docs.append(key)
                neg_scores.append(hits.at[idx_pred, "score"])

        data.append({
            "query": item["question"],
            "positives": {
                "doc_id": relevant_docs,
                "score": [1.0] * actual_positive
            },
            "negatives": {
                "doc_id": list(neg_docs),
                "score": neg_scores
            }
        })

    if not os.path.exists(os.path.join(data_dir,"bm25")):
        os.makedirs(os.path.join(data_dir,"bm25"))
    with open(os.path.join(data_dir, "bm25", f"bm25_{data_file.split('.')[0]}_top{top_k}.json"), "w") as file:
        json.dump(data, file)