# coding=utf-8

# Copyright 2018 The Google AI Language Team Authors, Allenai and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bert SQuAD evaluate."""
import re
import string
from collections import Counter, namedtuple, OrderedDict

from mxnet import nd

PredResult = namedtuple('PredResult', ['start', 'end'])

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i, _ in enumerate(index_and_score):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, tokenizer):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = OrderedDict()
        for (i, c) in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tok_text = ' '.join(tokenizer(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for i in tok_ns_to_s_map.keys():
        tok_index = tok_ns_to_s_map[i]
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def predict(features,
            results,
            tokenizer,
            max_answer_length=64,
            null_score_diff_threshold=0.0,
            n_best_size=10,
            version_2=False):
    """Get prediction results.

    Parameters
    ----------
    features : list of SQuADFeature
        List of squad features for the example.
    results : list of data.qa.PredResult
        List of model predictions for span start and span end.
    tokenizer: callable
        Tokenizer function.
    max_answer_length: int, default 64
        Maximum length of the answer tokens.
    null_score_diff_threshold: float, default 0.0
        If null_score - best_non_null is greater than the threshold predict null.
    n_best_size: int, default 10
        The total number of n-best predictions.
    version_2: bool, default False
        If true, the SQuAD examples contain some that do not have an answer.

    Returns
    -------
    prediction: str
        The final prediction.
    nbest : list of (str, float)
        n-best predictions with their probabilities.
    """

    _PrelimPrediction = namedtuple('PrelimPrediction',
                                   ['feature_index', 'start_index', 'end_index',
                                    'pred_start', 'pred_end'])

    _NbestPrediction = namedtuple(
        'NbestPrediction', ['text', 'pred_start', 'pred_end'])

    prelim_predictions = []
    score_diff = None

    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_pred_start = 0  # the start logit at the slice with min null score
    null_pred_end = 0  # the end logit at the slice with min null score

    for features_id, (result, feature) in enumerate(zip(results, features)):
        start_indexes = _get_best_indexes(result.start, n_best_size)
        end_indexes = _get_best_indexes(result.end, n_best_size)

        if version_2:
            feature_null_score = result.start[0] + \
                result.end[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = features_id
                null_pred_start = result.start[0]
                null_pred_end = result.end[0]

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=features_id,
   