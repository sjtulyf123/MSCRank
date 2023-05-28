import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LayerRNNCell
from tensorflow.python.layers.base import base_layer
from tensorflow.python.ops import math_ops, init_ops, array_ops, nn_ops
from tensorflow.python.ops.rnn import dynamic_rnn

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class NewRNNCell(LayerRNNCell):
    def __init__(self, num_units, xt_depth,list_depth,category_num=3, activation=None, reuse=None, kernel_initializer=None,
                 bias_initializer=None, name=None):
        super(NewRNNCell, self).__init__(_reuse=reuse, name=name)
        self.input_spec = base_layer.InputSpec(ndim=2)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self.category_num = category_num
        self.xt_depth = xt_depth
        self.list_depth = list_depth
        self.split_num=[0,self.xt_depth]
        for list_idx in range(self.category_num):
            self.split_num.append(self.split_num[-1]+self.list_depth)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        xt_depth = self.xt_depth
        l_depth = self.list_depth
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[xt_depth + self._num_units + self.category_num * l_depth, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._p_kernel = []
        self._p_bias = []
        for list_idx in range(self.category_num):
            _one_p_kernel = self.add_variable(
                "p_{}/{}".format(list_idx, _WEIGHTS_VARIABLE_NAME),
                shape=[xt_depth + l_depth, self._num_units],
                initializer=self._kernel_initializer)
            _one_p_bias = self.add_variable(
                "p_{}/{}".format(list_idx, _BIAS_VARIABLE_NAME),
                shape=[self._num_units],
                initializer=(
                    self._bias_initializer
                    if self._bias_initializer is not None
                    else init_ops.constant_initializer(1.0, dtype=self.dtype)))
            self._p_kernel.append(_one_p_kernel)
            self._p_bias.append(_one_p_bias)
        self.candidate_kernel = []
        self.candidate_bias = []

        for list_idx in range(self.category_num):
            _one_candidate_kernel = self.add_variable(
                "candidate_{}/{}".format(list_idx, _WEIGHTS_VARIABLE_NAME),
                shape=[xt_depth + l_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            _one_candidate_bias = self.add_variable(
                "candidate_{}/{}".format(list_idx, _BIAS_VARIABLE_NAME),
                shape=[self._num_units],
                initializer=(
                    self._bias_initializer
                    if self._bias_initializer is not None
                    else init_ops.constant_initializer(1.0, dtype=self.dtype)))
            self.candidate_kernel.append(_one_candidate_kernel)
            self.candidate_bias.append(_one_candidate_bias)
        self.built = True

    def call(self, inputs, state):
        x_t = inputs[:, self.split_num[0]:self.split_num[1]]
        list_inputs = []
        for list_idx in range(self.category_num):
            list_inputs.append(inputs[:, self.split_num[list_idx + 1]:self.split_num[list_idx + 2]])
        list_input_concat = array_ops.concat(list_inputs, axis=1)
        gate_inputs = math_ops.matmul(
            array_ops.concat([x_t, state, list_input_concat], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        p_inputs = []
        for list_idx in range(self.category_num):
            p_input = math_ops.matmul(
                array_ops.concat([x_t, list_inputs[list_idx]], 1), self._p_kernel[list_idx])
            p_input = nn_ops.bias_add(p_input, self._p_bias[list_idx])
            p_input = array_ops.expand_dims(p_input, 1)
            p_inputs.append(p_input)
        concat_p_input = array_ops.concat(p_inputs, axis=1)
        p = nn_ops.softmax(concat_p_input, axis=1)
        list_p = array_ops.split(value=p, num_or_size_splits=self.category_num, axis=1)

        candidates = []
        r_state = r * state
        for list_idx in range(self.category_num):
            _one_candidate_input = math_ops.matmul(array_ops.concat([x_t, r_state, list_inputs[list_idx]], 1),
                                                   self.candidate_kernel[list_idx])
            _one_candidate_input = nn_ops.bias_add(_one_candidate_input, self.candidate_bias[list_idx])
            _one_candidate_input = math_ops.mul(array_ops.squeeze(list_p[list_idx], 1), _one_candidate_input)
            candidates.append(_one_candidate_input)

        final_candidate = math_ops.add_n(candidates)
        c = self._activation(final_candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h
