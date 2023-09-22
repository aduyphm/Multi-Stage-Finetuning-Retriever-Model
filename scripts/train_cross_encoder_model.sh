#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoints/reranker_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/"
fi

mkdir -p "${OUTPUT_DIR}"

# For electra-large, learning rate > 1e-5 will lead to instability empirically
PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# deepspeed train_cross_encoder.py --deepspeed ds_config.json \
python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} train_cross_encoder.py \
    --model_name_or_path intfloat/multilingual-e5-small \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --fp16 \
    --seed 987 \
    --train_dir "bm25" \
    --corpus_file "viquad_corpus.json" \
    --rerank_max_length 192 \
    --rerank_use_rdrop True \
    --train_n_passages 20 \
    --rerank_forward_factor 4 \
    --dataloader_num_workers 1 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --warmup_steps 1000 \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 5 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model acc \
    --greater_is_better True \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"

