# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 16:34:03 2021

@author: xiaohuai le
"""

from tensorflow.keras.layers import Layer,RNN,GRUCell,Dense
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec

'''
A weight mask is added to the FC layer and the RNN layer for pruning.
We override Dense and GRU of TensorFlow
'''
class Dense_mask(Dense):
    
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
      
    super(Dense_mask, self).__init__(units,
               activation,
               use_bias,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               activity_regularizer,
               kernel_constraint,
               bias_constraint,
               **kwargs)

  def build(self, input_shape):
      
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    
    self.mask = self.add_weight(
        shape=[last_dim, 1],
        name='kernel_mask',
        initializer='ones',
        trainable = False)
    
    self.masked_kernel = self.kernel * self.mask
    
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    rank = len(inputs.shape)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.masked_kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = math_ops.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.masked_kernel)
      else:
        outputs = gen_math_ops.mat_mul(inputs, self.masked_kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

class GRUCell_mask(GRUCell):

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               reset_after=False,
               **kwargs):
    super(GRUCell_mask, self).__init__(units,
               activation,
               recurrent_activation,
               use_bias,
               kernel_initializer,
               recurrent_initializer,
               bias_initializer,
               kernel_regularizer,
               recurrent_regularizer,
               bias_regularizer,
               kernel_constraint,
               recurrent_constraint,
               bias_constraint,
               dropout,
               recurrent_dropout,
               implementation,
               reset_after,
               **kwargs)

  def build(self, input_shape):
    input_dim = input_shape[-1]
    
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)   
    
    self.kernel_mask = self.add_weight(
        shape=(1, self.units * 3),
        name='kernel_mask',
        initializer='ones',
        trainable = False)
    
    self.recurrent_mask_column = self.add_weight(
        shape=(1, self.units * 3),
        name='recurrent_mask_column',
        initializer='ones',
        trainable = False)
    
    self.recurrent_mask_row = self.add_weight(
        shape=(self.units, 1),
        name='recurrent_mask_row',
        initializer='ones',
        trainable = False)
    
    self.masked_kernel = self.kernel * self.kernel_mask
    self.masked_recurrent_kernel = self.recurrent_kernel * self.recurrent_mask_column * self.recurrent_mask_row
    
    if self.use_bias:
      if not self.reset_after:
        bias_shape = (3 * self.units,)
      else:
        # separate biases for input and recurrent kernels
        # Note: the shape is intentionally different from CuDNNGRU biases
        # `(2 * 3 * self.units,)`, so that we can distinguish the classes
        # when loading and converting saved weights.
        bias_shape = (2, 3 * self.units)
      self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
      
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = array_ops.unstack(self.bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = K.dot(inputs_z, self.masked_kernel[:, :self.units])
      x_r = K.dot(inputs_r, self.masked_kernel[:, self.units:self.units * 2])
      x_h = K.dot(inputs_h, self.masked_kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = K.bias_add(x_z, input_bias[:self.units])
        x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = K.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = K.dot(h_tm1_z, self.masked_recurrent_kernel[:, :self.units])
      recurrent_r = K.dot(h_tm1_r,
                          self.masked_recurrent_kernel[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = K.bias_add(recurrent_r,
                                 recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = K.dot(h_tm1_h, self.masked_recurrent_kernel[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1_h,
                            self.masked_recurrent_kernel[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = K.dot(inputs, self.masked_kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = K.bias_add(matrix_x, input_bias)

      x_z = matrix_x[:, :self.units]
      x_r = matrix_x[:, self.units: 2 * self.units]
      x_h = matrix_x[:, 2 * self.units:]

      if 0. < self.recurrent_dropout < 1.:
        h_tm1 = h_tm1 * rec_dp_mask[0]

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, self.masked_recurrent_kernel)
        if self.use_bias:
          matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = K.dot(h_tm1, self.masked_recurrent_kernel[:, :2 * self.units])

      recurrent_z = matrix_inner[:, :self.units]
      recurrent_r = matrix_inner[:, self.units:2 * self.units]

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * matrix_inner[:, 2 * self.units:]
      else:
        recurrent_h = K.dot(r * h_tm1,
                            self.masked_recurrent_kernel[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]


class GRU_mask(RNN):
  """Gated Recurrent Unit - Cho et al. 2014.

  There are two variants. The default one is based on 1406.1078v3 and
  has reset gate applied to hidden state before matrix multiplication. The
  other one is based on original 1406.1078v1 and has the order reversed.

  The second variant is compatible with CuDNNGRU (GPU-only) and allows
  inference on CPU. Thus it has separate biases for `kernel` and
  `recurrent_kernel`. Use `'reset_after'=True` and
  `recurrent_activation='sigmoid'`.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form. 
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before" (default),
      True = "after" (CuDNN compatible).

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = GRUCell_mask(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after,
        dtype=kwargs.get('dtype'))
    super(GRU_mask, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self.cell.reset_dropout_mask()
    self.cell.reset_recurrent_dropout_mask()
    return super(GRU_mask, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  @property
  def reset_after(self):
    return self.cell.reset_after

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    base_config = super(GRU_mask, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)
