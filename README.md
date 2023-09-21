# Multi Stages Finetuning Retriever Model
## Data Format

Raw dataset:
```bash
├── data
│   ├── title
│   └── paragraphs
│       ├── qas
│       │   ├── id
│       │   ├── question
│       │   └── ...
│       └── context
└── ...
```

Train dataset:
```bash
├── question
├── related_docs
│   └── id
└── ...
```

Corpus:
```bash
├── id
├── title
└── context
```

Triple dataset:
```bash
├── query
├── positives
│   ├── doc_id
│   └── score
└── negatives
    ├── doc_id
    └── score
```

## Stage 1: Train retriever with BM25 negatives
BM25 negatives mining
```
python bm25_negatives_mining.py \
        --data_dir data \
        --train_file "" \
        --data_file viquad.json \
        --corpus_file viquad_corpus.json \
        --top_k 20
```
Train retriever
```
# Train bi-encoder
bash scripts/train_biencoder_model.sh
```
If you have more than 1 corpus, please send them to folder `data/corpus` and merge them using: `ensemble_corpus.ipynb`

## Stage 2: Train with hard negatives
Hard negatives mining
```
python hard_negatives_mining.py \
        --data_dir data \
        --train_file viquad_train.json \
        --data_file viquad.json \
        --index_file viquad_index.index \
        --corpus_file viquad_corpus.json \
        --q_max_length 32 \
        --p_max_length 144 \
        --top_k 20
```
Train retriever
```
# Train bi-encoder
bash scripts/train_biencoder_model.sh
```
## Stage 3: Train with cross encoder


## Stage 4: Train with distillation model

