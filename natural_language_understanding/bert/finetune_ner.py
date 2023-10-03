#!/usr/bin/env python
#  coding: utf-8

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
"""Provides command-line interace for training BERT-based named entity recognition model."""

# coding: utf-8
import argparse
import logging
import random

import numpy as np
import mxnet as mx

import gluonnlp as nlp

from ner_utils import get_context, get_bert_model, dump_metadata, str2bool
from data.ner import BERTTaggingDataset, convert_arrays_to_text
from model.ner import BERTTagger, attach_prediction

# seqeval is a dependency that is specific to named entity recognition.
import seqeval.metrics


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description='Train a BERT-based named entity recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data file paths
    arg_parser.add_argument('--train-path', type=str, required=True,
                            help='Path to the training data file')
    arg_parser.add_argument('--dev-path', type=str, required=True,
                            help='Path to the development data file')
    arg_parser.add_argument('--test-path', type=str, required=True,
                            help='Path to the test data file')

    arg_parser.add_argument('--save-checkpoint-prefix', type=str, required=False, default=None,
                            help='Prefix of model checkpoint file')

    # bert options
    arg_parser.add_argument('--bert-model', type=str, default='bert_12_768_12',
                            help='Name of the BERT model')
    arg_parser.add_argument('--cased', type=str2bool, default=True,
                            help='Path to the development data file')
    arg_parser.add_argument('--dropout-prob', type=float, default=0.1,
                            help='Dropout probability for the last layer')

    # optimization p