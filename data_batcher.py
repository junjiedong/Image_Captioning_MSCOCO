# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training"""


import random
import time
import re

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID
from copy import deepcopy


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, image_features, caption_ids_input, caption_ids_label, caption_mask, image_id=None):
        """
        Inputs:
          image_features: Numpy arrays. Image feature extracted from images.
            Shape ().
          caption_ids_input: Numpy arrays. From SOS to last word in caption.
          caption_ids_input: Numpy arrays. From first word to EOS in caption.
          caption mask: Numpy arrays. Used to mask the caption.
          image_id: List of int. Corresponding image ids in this batch.
            Contains 1s where there is real data, 0s where there is padding.
        """
        self.image_features = image_features
        self.caption_ids_input = caption_ids_input
        self.caption_ids_label = caption_ids_label
        self.caption_mask = caption_mask
        self.image_id = image_id

        self.batch_size = len(self.image_features)


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids


def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max([len(x) for x in token_batch]) if batch_pad == 0 else batch_pad
    return [token_list + [PAD_ID] * (maxlen - len(token_list)) for token_list in token_batch]


def refill_batches(batches, word2id, image_features_map, caption_map, caption_image_map, caption_ids_list, batch_size, caption_len, start_index, data_source):
    """
    Adds more batches into the "batches" list. For training.

    Inputs:
      batches: list to add batches to
      word2id: dictionary mapping word (string) to word id (int)
      image_features_map: a map containing all images. Map from image id to features.
      caption_map: a map containing all the captions.
      caption_image_map: a map mapping caption id to image id.
      caption_ids_list: a list containing the caption_ids to refill from.
      batch_size: the batch size to to refill
      caption_len: max length of context and question respectively
      start_index: the index in caption ids. Refill from the start index.
    """
    print ("Refilling batches...")
    tic = time.time()
    examples = [] # list of (image_features, caption_ids_input, caption_ids_label, image_id) tuples

    index = start_index

    while index < len(caption_ids_list) and len(examples) < batch_size * 160: # while you haven't reached the end

        current_caption_id = caption_ids_list[index]
        current_image_id = caption_image_map[current_caption_id]

        if str(current_image_id) not in image_features_map:
            index += 1
            continue

        current_caption_string = caption_map[current_caption_id]
        # NOTE: CHANGE for SSD/RAM
        if data_source == "ram":
            current_image_features = image_features_map[str(current_image_id)]
        else:
            current_image_features = image_features_map[str(current_image_id)].value

        # Convert tokens to word ids
        _, caption_ids = sentence_to_token_ids(current_caption_string, word2id)


        # discard or truncate too-long questions
        if len(caption_ids) > caption_len + 1:
            caption_ids = caption_ids[:caption_len + 1]

        # add to examples
        # NOTE: Change
        caption_ids_input = caption_ids[:-1]
        caption_ids_label = caption_ids[1:]
        examples.append((current_image_features, caption_ids_input, caption_ids_label, current_image_id))

        index += 1

    # Once you've either got 160 batches or you've reached end of file:



    # Make into batches and append to the list batches
    for batch_start in range(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        image_features_batch, caption_ids_input_batch, caption_ids_label_batch, image_id_batch = list(zip(*examples[batch_start:batch_start+batch_size]))

        batches.append((image_features_batch, caption_ids_input_batch, caption_ids_label_batch, image_id_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print ("Refilling batches took %.2f seconds" % (toc-tic))
    return index



def refill_batches_eval(batches, image_features_map, image_ids_list, batch_size, start_index, data_source):
    """
    Adds more batches into the "batches" list, used in evaluation

    Inputs:
      batches: list to add batches to
      image_features_map: a map containing all images. Map from image id to features.
      image_ids_list: a list containing the image_ids to refill from.
      batch_size: the batch size to to refill
      start_index: the index in image_ids. Refill from the start index.
    """
    print ("Refilling batches...")
    tic = time.time()
    examples = [] # list of (image_features, caption_ids_input, caption_ids_label, image_id) tuples

    index = start_index

    while index < len(image_ids_list) and len(examples) < batch_size * 160: # while you haven't reached the end

        current_image_id = image_ids_list[index]

        if str(current_image_id) not in image_features_map:
            index += 1
            continue

        # NOTE: CHANGE for SSD/RAM
        if data_source == "ram":
            current_image_features = image_features_map[str(current_image_id)]
        else:
            current_image_features = image_features_map[str(current_image_id)].value

        examples.append((current_image_features, current_image_id))

        index += 1

    # Once you've either got 160 batches or you've reached end of file:



    # Make into batches and append to the list batches
    for batch_start in range(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        image_features_batch, image_id_batch = list(zip(*examples[batch_start:batch_start+batch_size]))

        batches.append((image_features_batch, image_id_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print ("Refilling batches took %.2f seconds" % (toc-tic))
    return index


def refill_batches_test(batches, image_features_list, batch_size, start_index):
    """
    Adds more batches into the "batches" list, used in test/prediction

    Inputs:
      batches: list to add batches to
      image_features_list: a list containing image features.
      batch_size: the batch size to to refill
      start_index: the index in image feature list. Refill from the start index.
    """
    print ("Refilling batches...")
    tic = time.time()
    examples = [] # list of (image_features, caption_ids_input, caption_ids_label, image_id) tuples

    index = start_index

    while index < len(image_features_list): # while you haven't reached the end

        current_image_features = image_features_list[index]

        examples.append((current_image_features, None))

        index += 1

        # stop refilling if you have 160 batches
        if len(examples) == batch_size * 160:
            break

    # Once you've either got 160 batches or you've reached end of file:



    # Make into batches and append to the list batches
    for batch_start in range(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        image_features_batch, image_id_batch = list(zip(*examples[batch_start:batch_start+batch_size]))

        batches.append((image_features_batch, image_id_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print ("Refilling batches took %.2f seconds" % (toc-tic))
    return index


def get_batch_generator(word2id, image_features_map, caption_map, caption_image_map, batch_size, caption_len, mode, image_features_list, data_source):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      image_features_map: a map containing all images. Map from image id to features.
      caption_map: a map containing all the captions.
      caption_image_map: a map mapping caption id to image id.
      batch_size: int. how big to make the batches
      caption_len: max length of context and question respectively
      train: If True, fill batch as in training.
    """

    caption_ids_list = None
    if mode == 'train':
        caption_ids_list = list(caption_map.keys())
        random.shuffle(caption_ids_list)

    image_ids = None
    # NOTE: Avoid storing duplicates in the image_ids list for 'eval' mode
    if mode == 'eval':
        image_ids = list({caption_image_map[k] for k in caption_map.keys()})

    batches = []

    start_index = 0

    while True:
        if len(batches) == 0: # add more batches
            if mode == 'train':
                start_index = refill_batches(batches, word2id, image_features_map, caption_map, caption_image_map,
                                             caption_ids_list, batch_size, caption_len, start_index, data_source)
            elif mode == 'eval':
                start_index = refill_batches_eval(batches, image_features_map, image_ids, batch_size, start_index, data_source)
            else:
                start_index = refill_batches_test(batches, image_features_list, batch_size, start_index)
        if len(batches) == 0:
            break

        # NOTE: CHANGE
        # Get next batch. These are all lists length batch_size
        if mode == 'train':
            (image_features, caption_ids_input, caption_ids_label, image_id) = batches.pop(0)

            # Pad context_ids and qn_ids
            caption_ids_input = padded(caption_ids_input, caption_len) # pad questions to length question_len
            caption_ids_label = padded(caption_ids_label, caption_len) # pad contexts to length context_len

            # Make qn_ids into a np array and create qn_mask
            caption_ids_input = np.array(caption_ids_input)
            caption_ids_label = np.array(caption_ids_label)

            caption_mask = (caption_ids_input != PAD_ID).astype(np.int32)

            if data_source != "ram":
                image_features = np.array(image_features)

            # Make into a Batch object
            batch = Batch(image_features, caption_ids_input, caption_ids_label, caption_mask, image_id)
        else:
            (image_features, image_id) = batches.pop(0)
            if data_source != "ram":
                image_features = np.array(image_features)
            batch = Batch(image_features, None, None, None, image_id)

        yield batch

    return
