
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
        print("Initializing the Caption Model...")
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id

        # Add all parts of the graph
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
        with vs.variable_scope("embeddings"):
            # Note: tf.constant means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix")

            # Get the word embeddings for the caption input
            self.caption_input_embs = tf.nn.embedding_lookup(embedding_matrix, self.caption_ids_input, name='caption_input_embs')


    def build_graph(self):
        """
        Builds the main part of the graph for the model.
        Defines:

        """
        pass


    def add_loss(self):
        """
        Add loss computation to the graph.
        Uses:
          self.logits: shape (batch_size, seq_len, vocab_size)
        Defines:
          self.loss: scalar tensor
        """
        # Use the 'weights' parameter to mask the output sequence
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.caption_ids_label, weights=self.caption_mask)


    def get_captions(self, session, batch):
        """
        Used for evaluation. Run forward-pass only; get the most likely answer span.
        Returns:
            captions: list of length batch_size, each element is a caption string
        """
        input_feed = {self.image_features: batch.image_features}  # Only need image_features for prediction
        output_feed = [ ]   # Whatever needed for prediction

        _ = session.run(output_feed, input_feed)

        # Decode the output

        return captions # {image_id: caption string}


    def get_val_loss(session):
        pass


    def check_val_metric(session, num_samples=0):
        '''
        Evaluate the model on the validation set.
        Inputs:
            num_samples: number of images to evaluate on. Evaluate on all val images if 0.
        '''
        captions = []  # [{"image_id": image_id, "caption": caption_str}]

        # Generate all the captions and save in list 'captions'

        # Dump the generated captions to json file
        file = open(self.FLAGS.train_res_dir, 'wb')
        json.dump(captions, file)
        file.close()

        # Evaluate using the official evaluation API (The evaluation takes ~12s for 1000 examples)
        cocoGold = COCO(self.FLAGS.goldAnn_val_dir) # Official annotations
        cocoRes = coco.loadRes(self.FLAGS.train_res_dir) # Prediction
        cocoEval = COCOEvalCap(cocoGold, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds() # Evaluate on a subset of the official captions_val2014
        cocoEval.evaluate()

        scores = cocoEval.eval  # {metric_name: metric_score}
        return scores   # Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr


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
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]
        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


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
            for batch in get_batch_generator(self.word2id, self.FLAGS.batch_size, self.FLAGS.max_caption_len):

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
                    val_scores = self.check_val_metric(session, num_samples=0)
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
