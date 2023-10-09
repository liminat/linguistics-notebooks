# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Utilities for pre-training."""
import glob
import time
import os
import functools
import logging
import argparse
import random
import multiprocessing

import numpy as np

import mxnet as mx
from mxnet.gluon.data import DataLoader
from create_pretraining_data import create_training_instances

import gluonnlp as nlp
from gluonnlp.data.batchify import Tuple, Stack, Pad
from gluonnlp.metric import MaskedAccuracy

__all__ = ['get_model_loss', 'get_pretrain_data_npz', 'get_dummy_dataloader',
           'save_parameters', 'save_states', 'evaluate', 'forward', 'split_and_load',
           'get_argparser', 'get_pretrain_data_text', 'generate_dev_set']

def get_model_loss(ctx, model, pretrained, dataset_name, vocab, dtype,
                   ckpt_dir=None, start_step=None):
    """Get model for pre-training.

    Parameters
    ----------
    ctx : Context or list of Context
        Contexts to initialize model
    model : str
        The name of the model, 'bert_12_768_12' or 'bert_24_1024_16'.
    pretrained : bool
        Whether to use pre-trained model weights as initialization.
    dataset_name : str
        The name of the dataset, which is used to retrieve the corresponding vocabulary file
        when the vocab argument is not provided. Options include 'book_corpus_wiki_en_uncased',
        'book_corpus_wiki_en_cased', 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
        'wiki_cn_cased'.
    vocab : BERTVocab or None
        The vocabulary for the model. If not provided, The vocabulary will be constructed
        based on dataset_name.
    dtype : float
        Data type of the model for training.
    ckpt_dir : str
        The path to the checkpoint directory.
    start_step : int or None
        If provided, it loads the model from the corresponding checkpoint from the ckpt_dir.

    Returns
    -------
    BERTModel : the model for pre-training.
    Loss : the next sentence prediction loss.
    Loss : the masked langauge model loss.
    BERTVocab : the vocabulary.
    """
    # model
    model, vocabulary = nlp.model.get_model(model, dataset_name=dataset_name, vocab=vocab,
                                            pretrained=pretrained, ctx=ctx)

    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.cast(dtype)

    if ckpt_dir and start_step:
        param_path = os.path.join(ckpt_dir, '%07d.params'%start_step)
        nlp.utils.load_parameters(model, param_path, ctx=ctx)
        logging.info('Loading step %d checkpoints from %s.', start_step, param_path)

    model.hybridize(static_alloc=True)

    # losses
    nsp_loss = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss = mx.gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True)
    mlm_loss.hybridize(static_alloc=True)

    return model, nsp_loss, mlm_loss, vocabulary

class BERTPretrainDataset(mx.gluon.data.ArrayDataset):
    """Dataset for BERT pre-training.

    Each record contains the following numpy ndarrays: input_ids, masked_lm_ids,
    masked_lm_positions, masked_lm_weights, next_sentence_labels, segment_ids, valid_lengths.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    tokenizer : BERTTokenizer
        The BERTTokenizer
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to use whole word masking.
    vocab : BERTVocab
        The BERTVocab
    num_workers : int
        The number of worker processes for dataset contruction.
    worker_pool : multiprocessing.Pool
        The worker process pool. Must be provided if num_workers > 1.
    """
    def __init__(self, filename, tokenizer, max_seq_length, short_seq_prob,
                 masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                 vocab, num_workers=1, worker_pool=None):
        logging.debug('start to load file %s ...', filename)
        dupe_factor = 1
        instances = create_training_instances(([filename], tokenizer, max_seq_length,
                                               short_seq_prob, masked_lm_prob,
                                               max_predictions_per_seq,
                                               whole_word_mask, vocab,
                                               dupe_factor, num_workers,
                                               worker_pool, None))
        super(BERTPretrainDataset, self).__init__(*instances)

def get_pretrain_data_text(data, batch_size, num_ctxes, shuffle, use_avg_len,
                           num_buckets, vocab, tokenizer, max_seq_length, short_seq_prob,
                           masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                           num_parts=1, part_idx=0,
                           prefetch=True, num_workers=1):
    """Get data iterators from raw text documents.

    Parameters
    ----------
    batch_size : int
        The batch size. If use_avg_len is set to True, batch_size is roughly the number of
        (non-padded) tokens in a batch.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : BERTVocab
        The vocabulary.
    tokenizer : BERTTokenizer or BERTSPTokenizer
        The tokenizer.
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs.
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to use whole word masking.
    n