# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:46:43 2021

@author: Xiaohuai Le
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import RNN
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest
import logging
from tensorflow.python.training.tracking import data_structures

def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = tf.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return tf.zeros(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)

def _binary_round(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    
    Based on http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html

    :param x: input tensor
    :return: y=round(x) with gradients defined by the identity mapping (y=x)
    """

    g = tf.get_default_graph()

    #with tf.name_scope("BinaryRound") as name:
    with g.gradient_override_map({"Round": "Identity"}):
        return tf.round(x)


class SkipGRUCell(Layer):

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
               implementation=1,
               reset_after=False,
               moving_ave=False,
               **kwargs):
    super(SkipGRUCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias
    self.moving_ave = moving_ave
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.implementation = implementation
    self.reset_after = reset_after
    self.state_size = self.units
    self.output_size = self.units
    self.linear = tf.keras.layers.Dense(1,bias_initializer='ones',activation='sigmoid')
    self.state_size = data_structures.NoDependency([self.units, 1, 1])
    
  def build(self, input_shape):
    input_dim = input_shape[-1]-1 
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
    '''
    Skip-GRU Cell based on the GRU implement of tensorflow
    
    inputs: the input of this time step
            the scale (gamma) of the update rate
            
    states: the hidden state,
            the update probability and the cumulative update probability of the last time step
    '''
    # GRU Cell
    h_tm1, update_prob_prev, cum_update_prob_prev = states[0],states[1],states[2]
    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = tf.unstack(self.bias)
    # scale is the gamma which used to control the update rate
    scale = inputs[:,-1:] 
    if self.implementation == 1:

      inputs_z = inputs[:,:-1]
      inputs_r = inputs[:,:-1]
      inputs_h = inputs[:,:-1]

      x_z = K.dot(inputs_z, self.kernel[:, :self.units])
      x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
      x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = K.bias_add(x_z, input_bias[:self.units])
        x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = K.bias_add(x_h, input_bias[self.units * 2:])

      h_tm1_z = h_tm1
      h_tm1_r = h_tm1
      h_tm1_h = h_tm1

      recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
      recurrent_r = K.dot(h_tm1_r,
                          self.recurrent_kernel[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = K.bias_add(recurrent_r,
                                 recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1_h,
                            self.recurrent_kernel[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:

      # inputs projected by all gate matrices at once
      matrix_x = K.dot(inputs[:,:-1], self.kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = K.bias_add(matrix_x, input_bias)

      x_z = matrix_x[:, :self.units]
      x_r = matrix_x[:, self.units: 2 * self.units]
      x_h = matrix_x[:, 2 * self.units:]

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
          matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])

      recurrent_z = matrix_inner[:, :self.units]
      recurrent_r = matrix_inner[:, self.units:2 * self.units]

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * matrix_inner[:, 2 * self.units:]
      else:
        recurrent_h = K.dot(r * h_tm1,
                            self.recurrent_kernel[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
      
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    
    # SKIP RNN
    new_update_prob_tilde = self.linear(h) * scale
    cum_update_prob = cum_update_prob_prev + tf.minimum(update_prob_prev, 1. - cum_update_prob_prev)
    update_gate = _binary_round(cum_update_prob)

    # Apply update gate
    if self.moving_ave:
        new_h = update_gate * h + (1. - update_gate) * (h_tm1 * 0.9 + self.activation(x_h) * 0.1)
    else:
        new_h = update_gate * h + (1. - update_gate) * h_tm1
    new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
    new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

    return [new_h,update_gate], [new_h, new_update_prob, new_cum_update_prob]  #tf.concat([h,new_update_prob,new_cum_update_prob],axis=-1)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'implementation': self.implementation,
        'reset_after': self.reset_after
    }
    base_config = super(SkipGRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  '''
  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))
  '''
class SkipGRU(RNN):

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
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               moving_ave=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = SkipGRUCell(
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
        implementation=implementation,
        reset_after=reset_after,
        moving_ave=moving_ave,
        dtype=kwargs.get('dtype'))
    super(SkipGRU, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    if initial_state is None:
        initial_state=[tf.zeros([tf.shape(inputs)[0],self.units]),tf.ones([tf.shape(inputs)[0],1]),tf.zeros([tf.shape(inputs)[0],1])]
    return super(SkipGRU, self).call(
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
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    base_config = super(SkipGRU, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)

  
if __name__ == '__main__':
    import numpy as np

    '''
    Test the Skip-GRU module on the MNIST data set
    '''
    inp = tf.keras.layers.Input(batch_shape = [100,None,29])
    rnn1,gate,h,p,cp =  SkipGRU(units = 64,
               return_sequences = True,return_state = True)(inp)
    results = tf.keras.layers.Dense(10,activation ='softmax')(h)
    
    gate_regular = tf.reduce_mean(gate) * 0.5
    model = tf.keras.models.Model(inp,results)


    def update_rate(y_true, y_pred):
        return tf.reduce_mean(gate)    
    
    from tensorflow.keras.datasets import mnist

    (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((-1, 28, 28))
    x_test = x_test.reshape((-1, 28,28))
    # inputs are padded with gamma = 1
    x_train = np.concatenate([x_train,np.ones([60000,28,1])],-1)
    x_test = np.concatenate([x_test,np.ones([10000,28,1])],-1)
    
    y_train = tf.keras.utils.to_categorical(y_train_)
    y_test = tf.keras.utils.to_categorical(y_test_)
    model.add_loss(gate_regular)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc', update_rate])
    
    model.fit(x_train,y_train,batch_size = 100,epochs =20)

