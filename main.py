
"""
This file contains the entrypoint to the rest of the code
The author is not very careful with absolute paths and absolute imports,
so only run this file in the main directory (./)
"""

import os
import io
import json
import sys
import logging
import tensorflow as tf

sys.path.append('coco/PythonAPI')  # For using COCO API
from model import CaptionModel
from vocab import get_glove

logging.basicConfig(level=logging.INFO)

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("data_source", "", "Whether to load all data into RAM (~50G). Available options: ram / ssd")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")
tf.app.flags.DEFINE_string("primary_metric", "CIDEr", "Primary evaluation metric. Use it for early stopping on the validation set.") # Bleu, METEOR, ROUGE_L, CIDEr

# Fixed (i.e. not intended to be tuned) Model Parameters
tf.app.flags.DEFINE_integer("embedding_size", 300, "Dimension of embeddings for words")
tf.app.flags.DEFINE_integer("max_caption_len", 19, "Maximum caption length (for both input and output)")
tf.app.flags.DEFINE_integer("image_dim1", 64, "Number of regions (eg. 8*8=64 for InceptionRes)")
tf.app.flags.DEFINE_integer("image_dim2", 1536, "Dimension of image feature for each region (eg. 1536 for InceptionRes)")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 512, "Size of the RNN states")
tf.app.flags.DEFINE_integer("beam_width", 3, "Beam width of beam search decoder")
tf.app.flags.DEFINE_string("special_token", "train", "Whether to make UNK and SOS trainable. Available options: zero/train")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 20, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 1000, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 1000, "How many iterations to do per evaluating on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# For eval mode
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For eval mode, which directory to load the checkpoint from. You need to specify this for eval mode.")

# Placeholders. Do not touch
tf.app.flags.DEFINE_string("MAIN_DIR", "", "_")
tf.app.flags.DEFINE_string("DATA_DIR", "", "_")
tf.app.flags.DEFINE_string("EXPERIMENTS_DIR", "", "_")
tf.app.flags.DEFINE_string("train_dir", "", "_")
tf.app.flags.DEFINE_string("bestmodel_dir", "", "_")
tf.app.flags.DEFINE_string("train_res_dir", "", "_")
tf.app.flags.DEFINE_string("glove_path", "", "_")
tf.app.flags.DEFINE_string("goldAnn_train_dir", "", "_")
tf.app.flags.DEFINE_string("goldAnn_val_dir", "", "_")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: CaptionModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print("Looking for model at %s..." % train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print("There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir)
            session.run(tf.global_variables_initializer())
            print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))


def main(unused_argv):
    # Check the supplied arguments
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)
    if not FLAGS.experiment_name:
        raise Exception("You need to specify --experiment_name")
    if not FLAGS.ckpt_load_dir and FLAGS.mode == "eval":
        raise Exception("You need to specify a directory to load the checkpoint for eval")
    if (not FLAGS.data_source) or (FLAGS.data_source != "ssd" and FLAGS.data_source != "ram"):
        raise Exception("You need to specify how to load data. Choose from ram and ssd.")

    FLAGS.MAIN_DIR = os.path.dirname(os.path.abspath(__file__))   # Absolute path of the directory containing main.py
    FLAGS.DATA_DIR = os.path.join(FLAGS.MAIN_DIR, "data")   # Absolute path of the data/ directory
    FLAGS.EXPERIMENTS_DIR = os.path.join(FLAGS.MAIN_DIR, "experiments")   # Absolute path of the experiments/ directory
    FLAGS.train_dir = os.path.join(FLAGS.EXPERIMENTS_DIR, FLAGS.experiment_name)
    FLAGS.bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")
    FLAGS.train_res_dir = os.path.join(FLAGS.train_dir, "myCaptions.json")  # Store the prediction results (for evaluation) during training

    FLAGS.glove_path = os.path.join(FLAGS.MAIN_DIR, "glove.6B.300d.trimmed.txt")
    FLAGS.goldAnn_train_dir = os.path.join(FLAGS.MAIN_DIR, "coco/annotations/captions_train2014.json")
    FLAGS.goldAnn_val_dir = os.path.join(FLAGS.MAIN_DIR, "coco/annotations/captions_val2014.json")

    # Load embedding matrix and vocab mappings
    random_init = (FLAGS.special_token == "train")
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, 300, random_init=random_init)

    # Initialize model
    caption_model = CaptionModel(FLAGS, id2word, word2id, emb_matrix)

    # Some GPU settings
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ####################################################################################
    ####################################################################################

    if FLAGS.mode == "train":
        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Make bestmodel dir if necessary
        if not os.path.exists(FLAGS.bestmodel_dir):
            os.makedirs(FLAGS.bestmodel_dir)

        with tf.Session(config=config) as sess:
            initialize_model(sess, caption_model, FLAGS.train_dir, expect_exists=False)  # Load most recent model
            caption_model.train(sess)

    ####################################################################################
    ####################################################################################

    # Sample evaluation command: python main.py --mode=eval --experiment_name=baseline --ckpt_load_dir=./experiments/baseline/best_checkpoint
    elif FLAGS.mode == "eval":
        print("Starting official evaluation...")
        with tf.Session(config=config) as sess:
            initialize_model(sess, caption_model, FLAGS.ckpt_load_dir, expect_exists=True)
            scores = caption_model.check_metric(sess, mode='val', num_samples=0)
            # Replace mode with 'test' if want to evaluate on test set
            for metric_name, metric_score in scores.items():
                print("{}: {}".format(metric_name, metric_score))

    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
