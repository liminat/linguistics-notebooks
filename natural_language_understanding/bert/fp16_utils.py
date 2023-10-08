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

"""Trainer for mixed precision training."""
import warnings
import collections
import numpy as np
import mxnet as mx
from mxnet import nd

def grad_global_norm(parameters, max_norm):
    """Calculate the 2-norm of gradients of parameters, and how much they should be scaled down
    such that their 2-norm does not exceed `max_norm`.

    If gradients exist for more than one context for a parameter, user needs to explicitly call
    ``trainer.allreduce_grads`` so that the gradients are summed first before calculating
    the 2-norm.

    .. note::

        This function is only for use when `update_on_kvstore` is set to False in trainer.

    Example::

        trainer = Trainer(net.collect_params(), update_on_kvstore=False, ...)
        for x, y in mx.gluon.utils.split_and_load(X, [mx.gpu(0), mx.gpu(1)]):
            with mx.autograd.record():
                y = net(x)
                loss = loss_fn(y, label)
            loss.backward()
        trainer.allreduce_grads()
        norm, ratio = grad_global_norm(net.collect_params().values(), max_norm)
        trainer.update(batch_size * ratio)
        ...

    Parameters
    ----------
    parameters : list of Parameters

    Returns
    -------
    NDArray
      Total norm. Shape is (1,)
    NDArray
      Ratio for rescaling gradients based on max_norm s.t. grad = grad / ratio.
      If total norm is NaN, ratio will be NaN, too. Shape is (1,)
    NDArray
      Whether the total norm is finite. Shape is (1,)
    """
    # collect gradient arrays
    arrays = []
    idx = 0
    for p in parameters:
        if p.grad_req != 'null':
            p_grads = p.list_grad()
            arrays.append(p_grads[idx % len(p_grads)])
            idx += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm.'

    # compute gradient norms
    def _norm(array):
        # TODO(haibin) norm operator does not support fp16 safe reduction.
        # Issue is tracked at: https://github.com/apache/incubator-mxnet/issues/14126
        x = array.reshape((-1,)).astype('float32', copy=False)
        return nd.dot(x, x)

    norm_arrays = [_norm(arr) for arr in arrays]

    # group norm arrays by ctx
    def group_by_ctx(arr_list):
        groups = collections.defaultdict(list)
        for arr in arr_list:
            ctx = arr.context
            groups[ctx].append(arr)
        return groups
    norm_groups = group_by_ctx(norm_arrays)

    # reduce
    ctx, dtype = arrays[0].context, 'float32'
    norms = [nd.add_n(*g).as_in_context(ctx) for g in norm_groups.values()]
    total_norm = nd.add_n(*norms).sqrt()
    scale = total_norm / max_norm
    # is_finite = 0 if NaN or Inf, 1 otherwise.
    is_finite = nd.contrib.isfinite(scale)
    # if scale is finite, nd.maximum selects the max between scale and 1. That is,
    # 1 is returned if total_norm does not exceed max_norm.
    # if scale = NaN or Inf, the result of nd.minimum is undefined. Therefore, we use
    # choices.take to return NaN or Inf.
    scale_or_one = nd.maximum(nd.ones((1,), dtype=dtype, ctx=ctx), scale)
    choices = nd.concat(scale, scale_or_one, dim=0)
    chosen_scale = choices.take(is_finite)
    return total_norm, chosen_scale, is_finite


class FP16Trainer(object):
    """ Trainer for mixed precision training.

    Parameters
    ----------
    trainer: gluon.Trainer
      the original gluon Trainer object for fp32 training.
    dynamic_loss_scale: bool. Default is True
      whether to use dynamic loss scaling. This is recommended for optimizing model
      parameters using FP16.
    loss_scaler_params : dict
        Key-word arguments to be passed to loss scaler constructor. For example,
        `{"init_scale" : 2.**15, "scale_window" : 2000, "tolerance" : 0.05}`
        for `DynamicLossScaler`.
        See each `LossScaler` for a list of supported arguments'
    """
    def __init__(self, trainer, dynamic_loss_scale=True, loss_scaler_params=None):
        if trainer._kvstore_params['update_on_kvstore'] is not False and trainer._kvstore:
            err = 'Only gluon.Trainer created with update_on_kvstore=False is supported.'
            raise NotImplementedError(err)
        self.fp32_trainer = trainer
        loss_scaler_params = loss_scaler_params if loss_scaler_params else {}
        self._scaler = DynamicLossScaler(**loss_scaler_params) if dynamic_loss_scale \
                       else StaticLossScaler(**loss_scaler_params)
        # if the optimizer supports NaN check, we can always defer the NaN check to the optimizer
        # TODO(haibin) this should be added via registry
        self._support_nan_check = trainer._optimizer.__class__.__name__ == 'BERTAdam'

    def backward(self, loss):
        """backward propagation with loss"""
        with mx.autograd.record():
            if isinstance(loss, (tuple, list)):
                ls = [l * self._scaler.loss_scale for l in loss]
            else:
                ls = loss * self._scaler.loss_scale
        mx.autograd.backward(ls)

    def step(self, batch_size, max_norm=None):
        """Makes one step of parameter update. Should be called after
        `fp16_optimizer.backward()`, and outside of `record()` scope.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        max_norm : NDArray, optional, default is None
            max value for global 2-norm of gradients.
        """
        self.fp32_trainer.allreduce_grads()
        step_size = batch_size * self._scaler.loss_scale
        if max_norm:
            norm, ratio, is_finite = grad_global_norm(self.fp32_trainer._params,
                                                      max_norm * self._scaler.loss_scale)
            step_size = ratio * step_size
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = is_finite.asscalar() < 1
            else:
                overflow = not np.isfinite(norm.asscalar())
                if not overflow:
                    self.fp32_trainer.update(step_size)
        else:
            # TODO(haibin) optimize the performance when max_norm is not present
            # sequentially adding isnan/isinf results may be slow
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
            else:
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
                if not overflow:
                    self.fp32_trainer.update(step_size)
        # update scale based on overflow information
        self._scaler.update_scale(overflow)

class LossScaler(object):
    """Abstract loss scaler"""
    de