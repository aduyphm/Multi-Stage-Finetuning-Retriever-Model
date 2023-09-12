import os
import random
import sys

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids

import pandas as pd
import json
from sklearn.model_selection import train_test_split



class RetrievalDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size > 0
        self.tokenizer = tokenizer
        
        documents_data = pd.read_csv(os.path.join(args.data_dir, 'vbpl.csv'), encoding='utf-16')
        self.corpus = Dataset.from_pandas(documents_data)
        
#         corpus_path = os.path.join(args.data_dir, 'passages.jsonl.gz')
#         self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)
        
        # doc_id = (so_hieu, dieu)
        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
            use_first_positive=self.args.use_first_positive
        )
        print(input_doc_ids)
        print(f"len(examples['query']): {len(examples['query'])}")
        print(f"train_n_passages: {self.args.train_n_passages}")
#         assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages
    
#         input_docs: List[str] = [self.corpus[doc_id]['noi_dung_vb'] for doc_id in input_doc_ids]
#         input_titles: List[str] = [self.corpus[doc_id]['prefix'] for doc_id in input_doc_ids]

        input_docs: List[str] = [self.corpus[self.corpus["so_hieu"] == doc_id["so_hieu"] and self.corpus["dieu"] == doc_id["dieu"]]['noi_dung_vb'] for doc_id in input_doc_ids]
        input_titles: List[str] = [self.corpus[self.corpus["so_hieu"] == doc_id["so_hieu"] and self.corpus["dieu"] == doc_id["dieu"]]['prefix'] for doc_id in input_doc_ids]

        query_batch_dict = self.tokenizer(examples['query'],
                                          max_length=self.args.q_max_len,
                                          padding=PaddingStrategy.DO_NOT_PAD,
                                          truncation=True)
        doc_batch_dict = self.tokenizer(input_titles,
                                        text_pair=input_docs,
                                        max_length=self.args.p_max_len,
                                        padding=PaddingStrategy.DO_NOT_PAD,
                                        truncation=True)

        merged_dict = {'q_{}'.format(k): v for k, v in query_batch_dict.items()}
        step_size = self.args.train_n_passages
        for k, v in doc_batch_dict.items():
            k = 'd_{}'.format(k)
            merged_dict[k] = []
            for idx in range(0, len(v), step_size):
                merged_dict[k].append(v[idx:(idx + step_size)])

        if self.args.do_kd_biencoder:
            qid_to_doc_id_to_score = {}

            def _update_qid_pid_score(q_id: str, ex: Dict):
                assert len(ex['doc_id']) == len(ex['score'])
                if q_id not in qid_to_doc_id_to_score:
                    qid_to_doc_id_to_score[q_id] = {}
                for doc_id, score in zip(ex['doc_id'], ex['score']):
                    qid_to_doc_id_to_score[q_id][int(doc_id)] = score

            for idx, query_id in enumerate(examples['query_id']):
                _update_qid_pid_score(query_id, examples['positives'][idx])
                _update_qid_pid_score(query_id, examples['negatives'][idx])

            merged_dict['kd_labels'] = []
            for idx in range(0, len(input_doc_ids), step_size):
                qid = examples['query_id'][idx // step_size]
                cur_kd_labels = [qid_to_doc_id_to_score[qid][doc_id] for doc_id in input_doc_ids[idx:idx + step_size]]
                merged_dict['kd_labels'].append(cur_kd_labels)
            assert len(merged_dict['kd_labels']) == len(examples['query_id']), \
                '{} != {}'.format(len(merged_dict['kd_labels']), len(examples['query_id']))

        # Custom formatting function must return a dict
        return merged_dict

    def _get_transformed_datasets(self) -> Tuple:
#         data_files = {}
#         if self.args.train_file is not None:
#             data_files["train"] = self.args.train_file.split(',')
#         if self.args.validation_file is not None:
#             data_files["validation"] = self.args.validation_file
#         raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

#         train_dataset, eval_dataset = None, None

        with open(os.path.join(self.args.data_dir, 'bm25_neg_pairs_top20.json'), 'r') as json_file:
            queries_data = json.load(json_file)
        queries_df = pd.DataFrame(queries_data)
        train_df, eval_df = train_test_split(queries_df, test_size=0.1, random_state=42)
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        train_dataset.set_transform(self._transform_func)
        eval_dataset.set_transform(self._transform_func)

#         if self.args.do_train:
#             if "train" not in raw_datasets:
#                 raise ValueError("--do_train requires a train dataset")
#             train_dataset = raw_datasets["train"]
#             if self.args.max_train_samples is not None:
#                 train_dataset = train_dataset.select(range(self.args.max_train_samples))
#             # Log a few random samples from the training set:
#             for index in random.sample(range(len(train_dataset)), 3):
#                 logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
#             train_dataset.set_transform(self._transform_func)

#         if self.args.do_eval:
#             if "validation" not in raw_datasets:
#                 raise ValueError("--do_eval requires a validation dataset")
#             eval_dataset = raw_datasets["validation"]
#             eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset
