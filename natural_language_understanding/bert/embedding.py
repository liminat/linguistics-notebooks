"""BERT embedding."""
import argparse
import io
import logging
import os

import numpy as np
import mxnet as mx

from mxnet.gluon.data import DataLoader

import gluonnlp
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from gluonnlp.base import get_home_dir

try:
    from data.embedding import BertEmbeddingDataset
except ImportError:
    from .data.embedding import BertEmbeddingDataset

try:
    unicode
except NameError:
    # Define `unicode` for Python3
    def unicode(s, *_):
        return s


def to_unicode(s):
    return unicode(s, 'utf-8')


__all__ = ['BertEmbedding']


logger = logging.getLogger(__name__)


class BertEmbedding(object):
    """
    Encoding from BERT model.

    Parameters
    ----------
    ctx : Context.
        running BertEmbedding on which gpu device id.
    dtype: str
        data type to use for the model.
    model : str, default bert_12_768_12.
        pre-trained BERT model
    dataset_name : str, default book_corpus_wiki_en_uncased.
        pre-trained model dataset
    params_path: str, default None
        path to a parameters file to load instead of the pretrained model.
    max_seq_length : int, default 25
        max length of each sequence
    batch_size : int, default 256
        batch size
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.
    """
    def __init__(self, ctx=mx.cpu(), dtype='float32', model='bert_12_768_12',
                 dataset_name='book_corpus_wiki_en_uncased', params_path=None,
                 max_seq_length=25, batch_size=256,
                 root=os.path.join(get_home_dir(), 'models')):
        self.ctx = ctx
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        # Don't download the pretrained models if we have a parameter path
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=self.dataset_name,
                                                         pretrained=params_path is None,
                                                         ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False,
                                                         root=root)
        self.bert.cast(self.dtype)

        if params_path:
            logger.info('Loading params from %s', params_path)
            self.bert.load_parameters(params_path, ctx=ctx, ignore_extra=True)

        lower = 'uncased' in self.dataset_name
        self.tokenizer = BERTTokenizer(self.vocab, lower=lower)
        self.transform = BERTSentenceTransform(tokenizer=self.tokenizer,
                                               max_seq_length=self.max_seq_length,
                                               pair=False)

    def __call__(self, sentences, oov_way='avg')