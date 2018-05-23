"""
This file contains the implementation of a custom SentinelLSTM cell
Majority of the implementation is borrowed from the official BasicLSTMCell
https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
"""

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base as base_layer

class SentinelLSTMCell(LayerRNNCell):
    """
    Based on TensorFlow built-in BasicLSTMCell
    The only difference is an additional sentinel gate and sentinel output
    Dropout is applied to the output within the cell
    """
    def __init__(self, num_units, output_keep_prob=1.0, forget_bias=1.0, reuse=None, name=None):
        super(SentinelLSTMCell, self).__init__(_reuse=reuse, name=name)
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self.output_keep_prob = output_keep_prob
        self._forget_bias = forget_bias
        self._activation = math_ops.tanh

    @property
    def state_size(self):
        return 2 * self._num_units  # state is not tuple

    @property
    def output_size(self):
        return 2 * self._num_units  # hidden state + sentinel vector

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable("kernel", shape=[input_depth + h_depth, 5 * self._num_units])
        self._bias = self.add_variable("bias", shape=[5 * self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        """
        inputs: tensor with shape [batch_size, input_size]
        state: tensor with shape [batch_size, 2 * self.state_size] (because state is concatenated)

        Returns:
            A pair containing the new hidden state, and the new concatenated state
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.

        # State is not tuple
        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate, s = sentinel_gate
        i, j, f, o, s = array_ops.split(value=gate_inputs, num_or_size_splits=5, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a  performance improvement
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                        multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))
        new_s = multiply(self._activation(new_c), sigmoid(s))   # sentinel output

        # State is not tuple
        new_state = array_ops.concat([new_c, new_h], 1)

        # Concatenate new_h and new_s to form the final output
        # Then apply dropout to the output before return
        output = array_ops.concat([new_h, new_s], 1)
        output = tf.nn.dropout(output, self.output_keep_prob)

        return output, new_state
