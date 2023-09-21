#!/usr/bin/env python
# coding: utf-8

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    BatchEncoding
)
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutput
from typing import List
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import torch
import pickle
import faiss
import os

def l2_normalize(x: torch.Tensor):
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

def encode_query(tokenizer: PreTrainedTokenizerFast, query: str) -> BatchEncoding:
    return tokenizer(query,
                     max_length=p_max_length,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')

def encode_passage(tokenizer: PreTrainedTokenizerFast, passage: str, title: str = "") -> BatchEncoding:
    return tokenizer(title,
                     text_pair=passage,
                     max_length=q_max_length,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="", type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--train_file", default="", type=str)
    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--corpus_file", default="", type=str)
    parser.add_argument("--index_file", default="", type=str)
    parser.add_argument("--q_max_length", default=32, type=int)
    parser.add_argument("--p_max_length", default=144, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    args = parser.parse_args()
    
    data_dir = os.path.join(os.getcwd(), args.data_dir)
    data_file = args.data_file
    train_file = args.train_file
    corpus_file = args.corpus_file
    index_file = args.index_file
    q_max_length = args.q_max_length
    p_max_length = args.p_max_length
    top_k = args.top_k
    
    prefix = data_file.split(".")[0]
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
                        "related_docs": [prefix + "_" + str(idx)]
                    })
                corpus.append({
                    "id": prefix + "_" + str(idx),
                    "title": title,
                    "context": paragraph["context"]
                })
                idx += 1
        with open(os.path.join(data_dir, corpus_file), 'w') as json_file:
            json.dump(corpus, json_file)
        with open(os.path.join(data_dir, train_file), 'w') as json_file:
            json.dump(train, json_file)
    else: 
        with open(os.path.join(data_dir, train_file), 'r') as json_file:
            train = json.load(json_file)
    
    # Read corpus
    if corpus_file.endswith(".json"):
        corpus_df = pd.read_json(os.path.join(data_dir, corpus_file))
    elif corpus_file.endswith(".csv"):
        corpus_df = pd.read_csv(os.path.join(data_dir, corpus_file), encoding="utf-16")
        
    # Load model
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Read index
    index_path = os.path.join(data_dir, index_file)
    if not os.path.exists(index_path):
        passages: List[str] = corpus_df['context'].astype(str).tolist()
        psg_embeddings = []
        for passage in tqdm(passages):
            psg_batch_dict = encode_passage(tokenizer, passage)
            outputs: BaseModelOutput = model(**psg_batch_dict, return_dict=True)
            psg_embeddings.append(l2_normalize(outputs.last_hidden_state[0, 0, :]))
#         norm_psg_embeddings = normalize_embeddings(psg_embeddings)
        index = faiss.IndexFlatIP(model.config.hidden_size)
        index.add(psg_embeddings)
        faiss.write_index(index, index_path)
    index = faiss.read_index(index_path)
    
    data = []
    for i, item in tqdm(enumerate(train)):
        query = item["question"]
        relevant_docs = item["related_docs"]
#         dict_relevant = {(article["so_hieu"], article["dieu"]) : article for article in relevant_articles}
        actual_positive = len(relevant_docs)

        query_batch_dict = encode_query(tokenizer, query)
        outputs: BaseModelOutput = model(**query_batch_dict, return_dict=True)
        query_embedding = l2_normalize(outputs.last_hidden_state[0, 0, :])
#         normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores, indices, embeddings = index.search_and_reconstruct(query_embedding, top_k)
        hits1 = corpus_df.iloc[indices[0]]
        hits = hits1.copy()
        hits["id"] = hits["id"].astype(str)
        hits["score"] = scores[0]

        neg_docs = []
        neg_scores = []
        for j, idx_pred in enumerate(hits.index):
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

    if not os.path.exists(os.path.join(data_dir,"hneg")):
        os.makedirs(os.path.join(data_dir,"hneg"))
    with open(os.path.join(data_dir, "hneg", f"hneg_{data_file.split('.')[0]}_top{top_k}.json"), "w") as file:
        json.dump(data, file)