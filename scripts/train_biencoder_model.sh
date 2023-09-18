#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoints/biencoder_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/"
fi

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} train_biencoder.py \
    --model_name_or_path intfloat/multilingual-e5-small \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --add_pooler False \
    --l2_normalize True \
    --t 0.02 \
    --seed 1234 \
    --do_train \
    --fp16 \
    --train_dir "bm25" \
    --corpus_file "viquad_corpus.json" \
    --q_max_len 32 \
    --p_max_len 144 \
    --train_n_passages 20 \
    --dataloader_num_workers 1 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --use_scaled_loss True \
    --warmup_steps 1000 \
    --share_encoder True \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 2 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"