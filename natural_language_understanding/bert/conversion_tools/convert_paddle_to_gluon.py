#!/usr/bin/env python
# encoding: utf-8
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import argparse
import logging
import json
import mxnet as mx
import gluonnlp as nlp
import paddle.fluid as fluid

from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams
from utils import get_hash, tf_vocab_to_gluon_vocab, load_text_vocab

parser = argparse.ArgumentParser()
parser.add_argument("--gluon_bert_model_base", default='ernie_12_768_12', type=str, help=".")
parser.add_argument("--init_pretraining_params", default='./ERNIE_stable-1.0.1/params',
                    type=str, help=".")
parser.add_argument("--ernie_config_path", default='./ERNIE_stable-1.0.1/ernie_config.json',
                    type=str, help=".")
parser.add_argument("--ernie_vocab_path", default='./ERNIE_stable-1.0.1/vocab.txt',
                    type=str, help=".")
parser.add_argument("--out_dir", default='./ernie_gluon_model2', type=str, help=".")
parser.add_argument("--baidu_lark_repo_dir", default='../../../../LARK', type=str,
                    help='path to the original baidu lark repository. '
                         'The repo should be at f97e3c8581e36dc1979560d62f75df862acd9585.'
                         '(https://github.com/PaddlePaddle/LARK.git)')
args = parser.parse_args()

sys.path = [os.path.join(args.baidu_lark_repo_dir,'ERNIE')] + sys.path
try:
    from model.ernie import ErnieConfig
    from finetune.classifier import create_model
except:
    raise ImportError('Place clone ERNIE first')

def if_exist(var):
    return os.path.exists(os.path.join(args.init_pretraining_params, var.name))


def build_weight_map():
    weight_map = collections.OrderedDict({
        'word_embedding': 'word_embed.0.weight',
        'pos_embedding': 'encoder.position_weight',
        'sent_embedding': 'token_type_embed.0.weight',
        'pre_encoder_layer_norm_scale': 'encoder.layer_norm.gamma',
        'pre_encoder_layer_norm_bias': 'encoder.layer_norm.beta',
    })

    def add_w_and_b(ernie_pre, gluon_pre):
        weight_map[ernie_pre + ".w_0"] = gluon_pre + ".weight"
        w