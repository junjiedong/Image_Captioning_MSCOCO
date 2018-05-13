"""
This file contains basic model components.
"""

import tensorflow as tf
import numpy as np

class RNNDecoder(object):
	"""
	General-purpose model to decode a sequence using a RNN.
	Use beam search for inference. Need to build training and inference graph separately. Share
	the same RNN cell and output layer with inference.
	Refer to the decoder in tensorflow NMT model: https://github.com/tensorflow/nmt
	"""
	def __init__(self,hidden_size,vocab_size,num_layers=1):
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
				initial_state = tf.contrib.seq2seq.tile_batch(initial_state,multiplier=beam_width)

				start_tokens = tf.fill([tf.shape(initial_state)[0]],infer_params['start_token'])
				# build beam search decoder
				beam_decoder =  tf.contrib.seq2seq.BeamSearchDecoder(
					cell=self.rnn_cell,
					embedding=infer_params['embedding'],
					start_tokens=start_tokens,
					end_token=infer_params['end_token'],
					initial_state=initial_state,
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
	def __init__(self,hidden_size):
		self.hidden_size = hidden_size

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
			return output
