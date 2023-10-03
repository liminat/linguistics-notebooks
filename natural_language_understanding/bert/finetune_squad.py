"""
SQuAD with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
SQuAD, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding=utf-8

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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import collections
import json
import logging
import os
import io
import copy
import random
import time
import warnings

import numpy as np
import mxnet as mx

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from model.qa import BertForQALoss, BertForQA
from data.qa import SQuADTransform, preprocess_dataset
from bert_qa_evaluate import get_F1_EM, predict, PredResult

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')


parser = argparse.ArgumentParser(description='BERT QA example.'
                                 'We fine-tune the BERT model on SQuAD dataset.')

parser.add_argument('--only_predict',
                    action='store_true',
                    help='Whether to predict only.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                    'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--pretrained_bert_parameters',
                    type=str,
                    default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                    ' default is ./output_dir')

parser.add_argument('--epochs',
                    type=int,
                    default=3,
                    help='number of epochs, default is 3')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer',
                    type=str,
                    default='bertadam',
                    help='optimization algorithm. defau