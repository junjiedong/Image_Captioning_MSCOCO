"""
This file contains basic model components.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape


class BasicAttentionLayer(base.Layer):
	"""
	This class implements a basic attention layer that is compatible with TensorFlow built-in decoder modules

	Inputs:
		LSTM hidden_state - shape [None, hidden_size] or [None, beam_width, hidden_size]
	Uses:
		Memory (i.e. image_features) - shape [None, image_dim1, hidden_size]
	Outputs:
		logits - shape [None, vocab_size] or [None, beam_width, vocab_size]
	"""

	def __init__(self, output_units, memory, keep_prob, beam_width, name=None, **kwargs):
		super(BasicAttentionLayer, self).__init__(trainable=True, name=name, activity_regularizer=None, **kwargs)
		self.output_units = output_units
		self.memory = memory
		self.keep_prob = keep_prob
		memory_shape = memory.get_shape().as_list()
		assert len(memory_shape) == 3
		self.image_dim1 = memory_shape[1]
		self.hidden_size = memory_shape[2]	# LSTM hidden size

		self.memory_tile = tf.tile(tf.expand_dims(memory, axis=1), [1, beam_width, 1, 1])
		assert self.memory_tile.get_shape().as_list() == [None, beam_width, self.image_dim1, self.hidden_size]
		self.memory_tile = tf.reshape(self.memory_tile, [-1, self.image_dim1, self.hidden_size])
		assert self.memory_tile.get_shape().as_list() == [None, self.image_dim1, self.hidden_size]

		self.attn_func = "trilinear"  # Options: trilinear / tanh

	def build(self, input_shape):
		"""
		Called once from __call__, when we know the shapes of of inputs and 'dtype'
		Should have the calls to add_variable()
		input_shape is of type 'TensorShape'
		"""
		input_shape = tensor_shape.TensorShape(input_shape)

		if self.attn_func == "trilinear":
			self.w_m = self.add_variable('kernel_attn_m', shape=[self.hidden_size], dtype=tf.float32, trainable=True)
			self.w_x = self.add_variable('kernel_attn_x', shape=[self.hidden_size], dtype=tf.float32, trainable=True)
			self.w_dot = self.add_variable('kernel_attn_dot', shape=[self.hidden_size], dtype=tf.float32, trainable=True)
		elif self.attn_func == "tanh":
			self.w_x = self.add_variable('kernel_attn_x', shape=[self.hidden_size, self.image_dim1], dtype=tf.float32, trainable=True)
			self.w_m = self.add_variable('kernel_attn_m', shape=[self.hidden_size, self.image_dim1], dtype=tf.float32, trainable=True)
			self.w_dot = self.add_variable('kernel_attn_dot', shape=[self.image_dim1], dtype=tf.float32, trainable=True)
		else:
			raise Exception("Invalid attention function option!")

		self.w_dense = self.add_variable('kernel_dense', shape=[3*self.hidden_size, self.hidden_size], dtype=tf.float32, trainable=True)
		self.b_dense = self.add_variable('bias_dense', shape=[self.hidden_size], initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=True)

		self.w_proj = self.add_variable('kernel_projection', shape=[self.hidden_size, self.output_units], dtype=tf.float32, trainable=True)

		self.built = True

	def call(self, inputs):
		"""
		Called in __call__ after making sure build() has been called once
		Should perform the logic of applying the layer to the input tensors
		"""
		shape = inputs.get_shape().as_list()
		assert shape[-1] == self.hidden_size
		beam = (len(shape) == 3)
		if beam:
			inputs = tf.reshape(inputs, [-1, self.hidden_size])

		# Just for convenience
		X = inputs
		M = self.memory_tile if beam else self.memory

		# Calculate attention similarity vector - shape (N, k)
		if self.attn_func == "trilinear":
			XM = tf.multiply(tf.expand_dims(X, 1), M)	# (N, k, h)
			X_logits = tf.tensordot(X, self.w_x, axes=[[1], [0]])  # (N,)
			M_logits = tf.tensordot(M, self.w_m, axes=[[2], [0]])  # (N, k)
			XM_logits = tf.tensordot(XM, self.w_dot, axes=[[2], [0]])  # (N, k)
			attn_logits = tf.expand_dims(X_logits, 1) + M_logits + XM_logits
			assert XM.get_shape().as_list() == [None, self.image_dim1, self.hidden_size]
			assert X_logits.get_shape().as_list() == [None]
			assert M_logits.get_shape().as_list() == [None, self.image_dim1]
			assert XM_logits.get_shape().as_list() == [None, self.image_dim1]
		elif self.attn_func == "tanh":
			X_logits = tf.matmul(X, self.w_x)  # (N, k)
			X_logits = tf.tile(tf.expand_dims(X_logits, 1), [1, self.image_dim1, 1])  # (N, k, k)
			M_logits = tf.tensordot(M, self.w_m, axes=[[2], [0]])  # (N, k, k)
			attn_logits = tf.tensordot(tf.tanh(X_logits + M_logits), self.w_dot, axes=[[2], [0]])  # (N, k)
			assert X_logits.get_shape().as_list() == [None, self.image_dim1, self.image_dim1]
			assert M_logits.get_shape().as_list() == [None, self.image_dim1, self.image_dim1]
		else:
			raise Exception("Invalid attention function option!")

		assert attn_logits.get_shape().as_list() == [None, self.image_dim1]
		attn_weights = tf.nn.softmax(attn_logits, axis=-1)
		attn_weights = tf.expand_dims(attn_weights, -1)  # (N, k, 1)
		attn_vec = tf.reduce_sum(attn_weights * M, axis=1)  # (N, h)
		assert attn_vec.get_shape().as_list() == [None, self.hidden_size]

		# Dense layer
		G = tf.concat([X, attn_vec, X * attn_vec], axis=-1)  # (N, 3*h)
		assert G.get_shape().as_list() == [None, 3*self.hidden_size]
		G = tf.nn.relu(tf.matmul(G, self.w_dense) + self.b_dense)
		G = tf.nn.dropout(G, self.keep_prob)
		assert G.get_shape().as_list() == [None, self.hidden_size]

		outputs = tf.matmul(G, self.w_proj)
		assert outputs.get_shape().as_list() == [None, self.output_units]

		if beam:
			outputs = tf.reshape(outputs, [-1, shape[1], self.output_units])

		self.attn_weights = attn_weights
		return outputs

	def compute_output_shape(self, input_shape):
		'''
		input_shape: A (possibly nested tuple of) TensorShape. It need not be fully defined (e.g. the batch size may be unknown).
		Returns: A (possibly nested tuple of) TensorShape
		'''
		input_shape = tensor_shape.TensorShape(input_shape)
		output_shape = input_shape[:-1].concatenate(self.output_units)
		return output_shape


class DenseLayer(base.Layer):
	"""
	Densely-connected layer class.
	This layer implements the operation: `outputs = inputs * kernel`
	`kernel` is a weights matrix created by the layer
	"""
	def __init__(self, units, memory_tensor, name=None, **kwargs):
		super(DenseLayer, self).__init__(trainable=True, name=name, activity_regularizer=None, **kwargs)
		self.units = units
		self.memory_tensor = memory_tensor

	def build(self, input_shape):
		"""
		Called once from __call__, when we know the shapes of of inputs and 'dtype'
		Should have the calls to add_variable()
		input_shape is of type 'TensorShape'
		"""
		input_shape = tensor_shape.TensorShape(input_shape)
		self.kernel = self.add_variable('kernel', shape=[input_shape[-1].value, self.units], dtype=tf.float32, trainable=True)
		self.built = True

	def call(self, inputs):
		"""
		Called in __call__ after making sure build() has been called once
		Should perform the logic of applying the layer to the input tensors
		"""
		shape = inputs.get_shape().as_list()
		beam = (len(shape) == 3)
		if beam:
			inputs = tf.reshape(inputs, [-1, shape[-1]])

		outputs = tf.matmul(inputs, self.kernel)
		if beam:
			outputs = tf.reshape(outputs, [-1, shape[1], self.units])
		return outputs

	def compute_output_shape(self, input_shape):
		'''
		input_shape: A (possibly nested tuple of) TensorShape. It need not be fully defined (e.g. the batch size may be unknown).
		Returns: A (possibly nested tuple of) TensorShape
		'''
		input_shape = tensor_shape.TensorShape(input_shape)
		output_shape = input_shape[:-1].concatenate(self.units)
		return output_shape



class RNNDecoder(object):
	"""
	General-purpose model to decode a sequence using a RNN.
	Use beam search for inference. Need to build training and inference graph separately. Share
	the same RNN cell and output layer with inference.
	Refer to the decoder in tensorflow NMT model: https://github.com/tensorflow/nmt
	"""
	def __init__(self, hidden_size, vocab_size, keep_prob, num_layers=1):
		"""
		Inputs:
			hidden_size: int. Hidden size of the RNN
			num_layers: int. Number of layers in RNN
		"""
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# self.rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
		# 	num_layers=self.num_layers,
		# 	num_units=hidden_size)
		self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=False)
		# NOTE: Add dropout
		self.rnn_cell = tf.contrib.rnn.DropoutWrapper(self.rnn_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=1.0)
		self.projection_layer = tf.layers.Dense(
			vocab_size,use_bias=False,name='output_projection')

	def build_graph(self,initial_state,decoder_inputs,masks,mode,infer_params=None):
		"""
		Inputs:
          	initial_state: Tensor shape (batch_size, hidden_size)
		  	decoder_inputs: Tensor shape (batch_size, max_len, vocab-size)
		  	masks: Tensor shape (batch_size, max_len)
            	Has 1s where there is real input, 0s where there's padding.
          	mode: "train" or "infer"
		  	infer_params: dictionary contains parameters for inference
		  		Dictionary should contain keys: beam_width, embedding, start_token,
		  										end_token, length_penalty_weight, maximum_length
		  		'embedding' is the same definition as params used in tf.nn.embedding_lookup
		  		'start_token' and 'end_token' are ids of the start/end token in embedding
		Outputs:
			(rnn_output,predicted_ids)
			rnn_output: Tensor shape (batch_size, max_len, vocab_size); None for 'infer' mode
			predicted_ids: Tensor shape (batch_size,?); None for 'train' mode
		"""
		with tf.variable_scope("decoder") as decoder_scope:
			# Build graph for training
			if mode == "train":
				# change masks to real sequence length of decoder inputs: Tensor shape (batch_size,)
				sequence_length = tf.reduce_sum(masks,axis=1)
				assert sequence_length.get_shape().as_list() == [None]
				# build decoder
				helper =  tf.contrib.seq2seq.TrainingHelper(decoder_inputs,sequence_length)
				basic_decoder =  tf.contrib.seq2seq.BasicDecoder(
					self.rnn_cell, helper, initial_state,
					output_layer = self.projection_layer)
				# dynamic decoding
				# possible options: impute_finished, maximum_iterations, swap_memory
				outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(basic_decoder,scope=decoder_scope)
				return outputs.rnn_output, None
			# Build graph for testing
			else:
				beam_width = infer_params['beam_width']
				# replicate initial_state and start_token
				initial_state_decoder = tf.contrib.seq2seq.tile_batch(initial_state,multiplier=beam_width)

				start_tokens = tf.fill([tf.shape(initial_state)[0]],infer_params['start_token'])
				# build beam search decoder
				beam_decoder =  tf.contrib.seq2seq.BeamSearchDecoder(
					cell=self.rnn_cell,
					embedding=infer_params['embedding'],
					start_tokens=start_tokens,
					end_token=infer_params['end_token'],
					initial_state=initial_state_decoder,
					beam_width=beam_width,
					output_layer = self.projection_layer,
					length_penalty_weight=infer_params['length_penalty_weight'])
				# dynamic decoding
				outputs,_, _= tf.contrib.seq2seq.dynamic_decode(
					beam_decoder,scope=decoder_scope,maximum_iterations=infer_params['maximum_length'])
				return None, outputs.predicted_ids

def masked_softmax(logits, mask):
    """
    Takes masked softmax over output of decoder
    Inputs:
      logits: Tensor shape (batch_size, max_len, embedding_size)
      mask: Tensor shape (batch_size, max_len)
        Has 1s where there's real data in logits, 0 where there's padding
    Returns:
      masked_logits: Tensor shape (batch_size, max_len, embedding_size)
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Tensor shape (batch_size, max_len, embedding_size)
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, 2) # apply softmax at embedding dimension
    return masked_logits, prob_dist

class BasicTransferLayer(object):
	"""
	Module to combine CNN with output decoder
	"""
	def __init__(self,hidden_size, keep_prob):
		self.hidden_size = hidden_size
		self.keep_prob = keep_prob	# keep_prob is a placeholder

	def build_graph(self,cnn_output):
		"""
		Inputs:
          	cnn_output: Tensor shape (batch_size, img_spatial_size, img_channel_size)
		Outputs:
			output: Tensor shape (batch_size, hidden_size)
		"""
		with tf.variable_scope("TransferLayer"):
			# take average over image spatial dimension
			fc_input = tf.reduce_mean(cnn_output,axis=1)
			# fully connected layer with default relu activation
			output = tf.contrib.layers.fully_connected(fc_input,self.hidden_size)
			assert output.get_shape().as_list() == [None, self.hidden_size]
			# NOTE: Add dropout
			output = tf.nn.dropout(output, self.keep_prob)

			return output
