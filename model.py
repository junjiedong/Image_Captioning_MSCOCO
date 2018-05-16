
"""This file defines the top-level model"""

import time
import logging
import os
import sys
import numpy as np
import tensorflow as tf

from pycocotools.coco import COCO
from coco.coco_caption_py3.pycocoevalcap.eval import COCOEvalCap
from data_batcher import get_batch_generator
from modules import BasicTransferLayer, RNNDecoder
from vocab import PAD_ID, UNK_ID, EOS_ID, SOS_ID    # 0, 1, 2, 3
import h5py
import _pickle as cPickle
import json


class CaptionModel(object):
    """Top-level Image Captioning module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the image captioning model.
        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (vocab_size, embedding_size) containing pre-traing GloVe embeddings
        """
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.vocab_size = len(id2word)  # Number of words in the vocabulary

        print("Reading dataset metadata...");
        self.caption_id_2_img_id = cPickle.load(open(os.path.join(FLAGS.DATA_DIR, "caption_id_2_img_id.p"), 'rb'))
        self.train_caption_id_2_caption = cPickle.load(open(os.path.join(FLAGS.DATA_DIR, "train_caption_id_2_caption.p"), 'rb'))
        self.val_caption_id_2_caption = cPickle.load(open(os.path.join(FLAGS.DATA_DIR, "val_caption_id_2_caption.p"), 'rb'))
        self.test_caption_id_2_caption = cPickle.load(open(os.path.join(FLAGS.DATA_DIR, "test_caption_id_2_caption.p"), 'rb'))

        # Load the hdf5 file (Always load val set; train/test depends)
        load_test = (FLAGS.mode == "eval")   # Whether to load in the test data
        test_img_set = {str(self.caption_id_2_img_id[cpid]) for cpid in self.test_caption_id_2_caption}
        train_img_set = {str(self.caption_id_2_img_id[cpid]) for cpid in self.train_caption_id_2_caption}
        print("Number of images in training set: {}".format(len(train_img_set)))
        print("Number of images in test set: {}".format(len(test_img_set)))

        timg_features_map = h5py.File('./data/img_features.hdf5', 'r')
        if FLAGS.data_source == "ssd":
            print("Data will be loaded from SSD during training.")
            self.img_features_map = timg_features_map
        else:
            print("Start loading all data into RAM. Be patient!")
            self.img_features_map = {}
            num_dumped_test = 0
            for i, k in enumerate(timg_features_map.keys()):
                if FLAGS.mode == "eval" and k in train_img_set: # Don't load the training data in eval mode
                    continue
                if load_test or (k not in test_img_set):
                    self.img_features_map[k] = np.array(timg_features_map[k])
                if k in test_img_set:
                    num_dumped_test += 1
                if i % 100 == 0:
                    print("{} images processed...".format(i))  # printed i off by one, don't care though
            print("Finished loading all the requested data into RAM.")
            if FLAGS.mode == "eval":
                print("Did not load in training set data")
            if not load_test:
                print("Did not load in data for the {} test set images".format(num_dumped_test))

        # Add all parts of the graph
        print("Initializing the Caption Model...")
        with tf.variable_scope("CaptionModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)):
            # Use He Initialization by default
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs. These are all batch-first: the None corresponds to batch_size.
        self.image_features = tf.placeholder(tf.float32, shape=[None, self.FLAGS.image_dim1, self.FLAGS.image_dim2], name='image_features')
        self.caption_ids_input = tf.placeholder(tf.int32, shape=[None, self.FLAGS.max_caption_len]) # (<SOS> -> last_word) + PAD
        self.caption_ids_label = tf.placeholder(tf.int32, shape=[None, self.FLAGS.max_caption_len]) # (<first_word> -> <EOS>) + PAD
        self.caption_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.max_caption_len], name='caption_mask')  # Shared by the input and the label

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.
        Inputs:
          emb_matrix: The GloVe vectors, plus vectors for <SOS>, <UNK>, and <PAD>. Shape (vocab_size, embedding_size=300).
        """
        with tf.variable_scope("embeddings"):
            if self.FLAGS.special_token == "train":
                trainable_emb = tf.Variable(initial_value=emb_matrix[:4,:], trainable=True, name='trainable_emb_matrix', dtype=tf.float32)
                constant_emb = tf.constant(emb_matrix[4:,:], dtype=tf.float32, name="constant_emb_matrix")
                self.embedding_matrix = tf.concat(values=[trainable_emb, constant_emb], axis=0)
            else:
                # Note: tf.constant means it's not a trainable parameter
                self.embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix")

            # Get the word embeddings for the caption input
            self.caption_input_embs = tf.nn.embedding_lookup(self.embedding_matrix, self.caption_ids_input, name='caption_input_embs')


    def build_graph(self):
        """
        Builds the main part of the graph the model.
        Defines:
            self.logits: output from decoder, used for training. Shape (batch_size, T, vocab_size)
            self.predicted_ids: output ids from decoder, used for evaluation. Shape (batch_size, T, beam_width)
        """
        # Use fully connected layer to transfer output of cnn
        self.transfer_layer = BasicTransferLayer(2 * self.FLAGS.hidden_size, self.keep_prob)
        decoder_initial_state = self.transfer_layer.build_graph(self.image_features)
        # Use LSTM to decode the caption
        self.decoder = RNNDecoder(self.FLAGS.hidden_size, self.vocab_size, self.keep_prob)
        # build graph for training
        decoder_output,_ = self.decoder.build_graph(
            decoder_initial_state,self.caption_input_embs,self.caption_mask,"train")

        assert decoder_output.get_shape().as_list() == [None, None, self.vocab_size]

        # build graph for inferring
        infer_params={'embedding':self.embedding_matrix, 'start_token':SOS_ID, 'end_token':EOS_ID, 'length_penalty_weight':0.0}
        infer_params['beam_width']=self.FLAGS.beam_width
        infer_params['maximum_length']=self.FLAGS.max_caption_len
        _,predicted_ids = self.decoder.build_graph(
            decoder_initial_state,self.caption_input_embs,self.caption_mask,"infer",infer_params)

        assert predicted_ids.get_shape().as_list() == [None, None, self.FLAGS.beam_width]


        self.logits = decoder_output
        self.predicted_ids = predicted_ids


    def add_loss(self):
        """
        Add loss computation to the graph.
        Uses:
          self.logits: shape (batch_size, seq_len, vocab_size)
        Defines:
          self.loss: scalar tensor (averaged across both time and batch)
        """
        # Use the 'weights' parameter to mask the output sequence
        padding = [[0, 0], [0, self.FLAGS.max_caption_len - tf.shape(self.logits)[1]], [0, 0]]
        self.logits = tf.pad(self.logits, padding, "CONSTANT")
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.caption_ids_label, weights=tf.cast(self.caption_mask, tf.float32))


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        input_feed = {}
        input_feed[self.image_features] = batch.image_features
        input_feed[self.caption_ids_input] = batch.caption_ids_input
        input_feed[self.caption_ids_label] = batch.caption_ids_label
        input_feed[self.caption_mask] = batch.caption_mask
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        # output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]
        output_feed = [self.updates, self.loss, self.global_step, self.param_norm, self.gradient_norm]
        [_, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        # summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        '''
        Evaluate the loss on a batch of input. Forward-pass only.
        Needs:
            batch.image_features
            batch.caption_ids_input
            batch.caption_ids_label
            batch.caption_mask
        Returns:
            loss: The loss (averaged across the batch) for this batch
        '''

        input_feed = {} # Do not supply keep_prob here so it will default to 1
        input_feed[self.image_features] = batch.image_features
        input_feed[self.caption_ids_input] = batch.caption_ids_input
        input_feed[self.caption_ids_label] = batch.caption_ids_label
        input_feed[self.caption_mask] = batch.caption_mask

        output_feed = [self.loss]
        [loss] = session.run(output_feed, input_feed)
        return loss


    def get_captions(self, session, batch):
        """
        Used for evaluation. Run forward-pass on the batch.
        Needs：
            batch.image_id
            batch.image_features
        Returns:
            captions: {imgae_id: caption_string} (size of batch.size)
        """
        # Only need image_features for prediction. Do not supply keep_prob here so it will default to 1
        input_feed = {self.image_features: batch.image_features}
        output_feed = [self.predicted_ids]
        [predicted_ids] = session.run(output_feed, input_feed)  # (batch_size, max_len, beam_width)
        predicted_ids = predicted_ids[:, :, 0]  # Only take the best result for each one

        # Decode the output
        captions = {}
        for i, pred in enumerate(predicted_ids): # For each example in the batch, shape (max_len,)
            tokens = []
            for wid in pred:
                if wid == EOS_ID:
                    break
                tokens.append(self.id2word[wid])

            captions[batch.image_id[i]] = " ".join(tokens)

        return captions


    def get_val_loss(self, session):
        '''
        Get average loss on the entire val set
        This function is called periodically during training
        '''
        total_loss, num_examples = 0., 0
        tic = time.time()
        for batch in get_batch_generator(self.word2id, self.img_features_map, self.val_caption_id_2_caption, self.caption_id_2_img_id, \
                                        self.FLAGS.batch_size, self.FLAGS.max_caption_len, 'train', None, self.FLAGS.data_source):
            total_loss += self.get_loss(session, batch) * batch.batch_size
            num_examples += batch.batch_size

        logging.info("Computing validation loss over {} examples took {} seconds".format(num_examples, time.time() - tic))
        return total_loss / num_examples


    def check_metric(self, session, mode='val', num_samples=0):
        '''
        Evaluate the model on the validation or test set.
        Inputs:
            mode: should be either 'val' or 'test'
            num_samples: number of images to evaluate on. Evaluate on all val images if 0.
        '''
        assert (mode == 'val' or mode == 'test')
        captions = []  # [{"image_id": image_id, "caption": caption_str}]

        # Generate all the captions and save in list 'captions'
        tic = time.time()
        num_seen = 0  # Record the number of samples predicted so far
        this_caption_map = self.val_caption_id_2_caption if mode == 'val' else self.test_caption_id_2_caption

        for batch in get_batch_generator(self.word2id, self.img_features_map, this_caption_map, self.caption_id_2_img_id, \
                                        self.FLAGS.batch_size, self.FLAGS.max_caption_len, 'eval', None, self.FLAGS.data_source):
            batch_captions = self.get_captions(session, batch)   # {imgae_id: caption_string}
            for id, cap in batch_captions.items():
                captions.append({"image_id": id, "caption": cap})

            num_seen += batch.batch_size
            if num_samples != 0 and num_seen >= num_samples:
                break

        logging.info("Predicting on {} examples took {} seconds".format(num_seen, time.time() - tic))

        # Dump the generated captions to json file
        file = open(self.FLAGS.train_res_dir, 'w')
        json.dump(captions, file)
        file.close()

        # Evaluate using the official evaluation API (The evaluation takes ~12s for 1000 examples)
        tic = time.time()
        cocoGold = COCO(self.FLAGS.goldAnn_val_dir) # Official annotations
        cocoRes = cocoGold.loadRes(self.FLAGS.train_res_dir) # Prediction
        cocoEval = COCOEvalCap(cocoGold, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds() # Evaluate on a subset of the official captions_val2014
        cocoEval.evaluate()
        logging.info("Evaluating {} predictions took {} seconds".format(num_seen, time.time() - tic))

        scores = cocoEval.eval  # {metric_name: metric_score}
        return scores   # Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr


    def train(self, session):
        """
        Main training loop.
        """
        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management. We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "latest.ckpt")
        bestmodel_dir = self.FLAGS.bestmodel_dir
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "best.ckpt")
        best_val_metric = None

        # For TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, self.img_features_map, self.train_caption_id_2_caption,
                                             self.caption_id_2_img_id, self.FLAGS.batch_size, self.FLAGS.max_caption_len, 'train', None, self.FLAGS.data_source):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))
                    write_summary(loss, "train/loss", summary_writer, global_step)

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate the model
                if global_step % self.FLAGS.eval_every == 0:
                    # Get loss for entire val set and log to tensorboard
                    val_loss = self.get_val_loss(session)
                    logging.info("Epoch %d, Iter %d, Val loss: %f" % (epoch, global_step, val_loss))
                    write_summary(val_loss, "val/loss", summary_writer, global_step)

                    # Evaluate on val set and log all the metrics to tensorboard
                    val_scores = self.check_metric(session, mode='val', num_samples=0)
                    val_metric = val_scores[self.FLAGS.primary_metric]
                    for metric_name, metric_score in val_scores.items():
                        logging.info("Epoch {}, Iter {}, Val {}: {}".format(epoch, global_step, metric_name, metric_score))
                        write_summary(metric_score, "val/"+metric_name, summary_writer, global_step)

                    # Early stopping based on val evaluation
                    if best_val_metric is None or val_metric > best_val_metric:
                        best_val_metric = val_metric
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)

            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

# End of file
