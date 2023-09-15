# Multi Stages Finetuning Retriever Model
## Data Format

Not law dataset:
```bash
├── data
│   ├── title
│   └── paragraphs
│       ├── qas
│       │   ├── question
│       │   └── ...
│       └── context
└── ...
```

Law dataset:
```bash
├── question
├── related_docs
│   ├── no
│   └── articles
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

## Stage 1: Train with BM25 negatives
BM25 negatives mining
```
python bm25_negatives_mining.py \
        --data_dir data \
        --data_file viquad.json \
        --corpus_file viquad_corpus.json \
        --top_k 20
```

## Stage 2: Train with hard negatives
Hard negatives mining


## Stage 3: Train with cross encoder


## Stage 4: Train with distillation model

