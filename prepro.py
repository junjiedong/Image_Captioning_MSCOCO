'''
This file splits the official Coco validation set to val/test/extra_train, then processes
all the captions to generate the following mappings for later use:

caption_id_2_img_id
train_caption_id_2_caption
val_caption_id_2_caption
test_caption_id_2_caption
length_2_caption_id
length_2_num_captions

This file tokenizes the captions, replaces infrequent words and out-of-vocab words
with a special <UNK> token, prepends <SOS> token, and appends <EOS> token.

This file generates a trimmed GloVe file that contains only frequently used words

This file also generates file 'coco/coco_raw.json', which is a densed summary of the entire dataset

All the generated logs are saved to file "./prepro_log.txt"
'''

import os
import sys
import argparse
import json
import numpy as np
import re
import random
import logging
import _pickle as cPickle
from utils import tokenize

curr_path = os.path.abspath('./')
api_path = os.path.join(curr_path, 'coco/PythonAPI')
sys.path.append(api_path)
from pycocotools.coco import COCO

def generate_short_json(captions_dir):
    '''
    Generate a json summarizing useful information of the entire dataset (official train + official val)
    This file is useful for exploring the dataset and showing images (since it contains paths of images)
    File 'coco_raw.json' is saved to folder 'coco/'
    Below is an example entry of the data file:

    {'file_path': 'val/COCO_val2014_000000391895.jpg',
    'id': 391895,
    'captions': ['A man with a red helmet on a small moped on a dirt road. ',
                'Man riding a motor bike on a dirt road on the countryside.',
                'A man riding on the back of a motorcycle.',
                'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains.p',
                'A man in a red shirt and a red hat is on a motorcycle on a hill side.']}
    '''
    logging.info("Start: Generate short json file.")
    train_ann_path = os.path.join(captions_dir, 'captions_train2014.json')
    val_ann_path = os.path.join(captions_dir, 'captions_val2014.json')
    output_path = "data/coco_raw.json"

    train = json.load(open(train_ann_path, 'r'))
    val = json.load(open(val_ann_path, 'r'))

    # combine all images and annotations together
    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    # Group annotations by image
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if imgid not in itoa:
            itoa[imgid] = []
        itoa[imgid].append(a)

    # Create the json blob
    out = []
    for i, img in enumerate(imgs):
        imgid = img['id']
        loc = 'train' if 'train' in img['file_name'] else 'val'
        jimg = {}
        jimg['file_path'] = os.path.join(loc, img['file_name'])
        jimg['id'] = imgid
        jimg['captions'] = [a['caption'] for a in itoa[imgid]]

        out.append(jimg)

    json.dump(out, open(output_path, 'w'))
    logging.info("Finished: Generate short json file.")


def split_data(img_ids, params):
    '''
    Does:
        Split the official validation set images into three sets.
        After reserving the validation set and test set, assign the rest of the
        official validation set to the training set
    Returns:
        val_img_ids, test_img_ids, extra_train_img_ids (list of ints)
    '''
    num_val, num_test = params['num_val'], params['num_test']
    assert (num_val + num_test) <= len(img_ids)
    random.shuffle(img_ids)
    logging.info("Shuffled the official validation set, and performed train/val/test split.")
    return img_ids[:num_val], img_ids[num_val:num_val+num_test], img_ids[num_val+num_test:]

def generate_id_maps(coco_train, coco_val, val_img_set, test_img_set):
    '''
    Returns:
        caption_id_2_img_id: {caption_id: img_id} for all captions
        train_caption_id_2_caption : {caption_id: tokenized_caption_string}
        val_caption_id_2_caption
        test_caption_id_2_caption
    '''
    def genmap_helper(coco, progress_count, caption_count, truncate_count):
        for img_id in coco.getImgIds():
            caption_ids = coco.getAnnIds(imgIds=img_id)
            caption_objs = coco.loadAnns(caption_ids)
            for caption_obj in caption_objs:
                caption_id, caption_seq = caption_obj['id'], caption_obj['caption']
                tokenized_caption = tokenize(caption_seq)
                if len(tokenized_caption) > params['max_length']:
                    truncate_count += 1
                    tokenized_caption = tokenized_caption[:params['max_length']] # Truncate

                tokenized_caption = " ".join(tokenized_caption) # Convert to string
                caption_id_2_img_id[caption_id] = img_id
                if img_id in val_img_set:
                    val_caption_id_2_caption[caption_id] = tokenized_caption
                elif img_id in test_img_set:
                    test_caption_id_2_caption[caption_id] = tokenized_caption
                else:   # Must be in training set
                    train_caption_id_2_caption[caption_id] = tokenized_caption

                caption_count += 1

            progress_count += 1
            if progress_count % 1000 == 0:
                logging.info("Finished processing captions for {} images".format(progress_count))

        return progress_count, caption_count, truncate_count

    caption_id_2_img_id = {}
    train_caption_id_2_caption = {}
    val_caption_id_2_caption = {}
    test_caption_id_2_caption = {}

    progress_count = 0  # Number of images processed
    caption_count = 0   # Number of captions processed
    truncate_count = 0  # Number of truncated captions
    logging.info("Start: Tokenize captions and generate caption maps.")
    progress_count, caption_count, truncate_count = genmap_helper(coco_train, progress_count, caption_count, truncate_count)
    progress_count, caption_count, truncate_count = genmap_helper(coco_val, progress_count, caption_count, truncate_count)
    logging.info("Finished: Tokenize captions and generate caption maps.")
    logging.info("{} of the {} captions exceeded maximum length, and were therefore truncated.".format(truncate_count, caption_count))
    return caption_id_2_img_id, train_caption_id_2_caption, val_caption_id_2_caption, test_caption_id_2_caption


def generate_vocab(glove_path, trimmed_path, train_caption_id_2_caption):
    '''
    Does:
        Calculate word counts and log statistics
        Write the trimmed glove matrix to file |trimmed_path|
    Returns:
        The final vocabulary to be used for both training and inference
    '''
    word_count_threshold = params['word_count_threshold']

    logging.info("Start: Word Count (only use new training set for convenience)")
    word_counts = {}    # {word: count}
    for id, caption in train_caption_id_2_caption.items():
        for w in caption.split(" "):
            word_counts[w] = word_counts.get(w, 0) + 1
    total_word_count = sum(word_counts.values())    # Total number of words
    logging.info("Total number of words in training set: {}".format(total_word_count))

    logging.info("Sanity check: Print most frequent words.")
    for w in sorted(word_counts.keys(), key=lambda k: word_counts[k], reverse=True)[:10]:
        logging.info("{}: {}".format(w, word_counts[w]))

    vocab = []   # The final vocabulary
    with open(glove_path, 'r') as glove:
        with open(trimmed_path, 'w') as trimmed:
            for line in glove:
                w = line.split(" ")[0]
                if word_counts.get(w, 0) >= word_count_threshold:
                    vocab.append(w)
                    trimmed.write(line)

    logging.info("Final vocabulary size: {}".format(len(vocab)))
    logging.info("Finished writing the trimmed GloVe file.")
    return vocab


def modify_captions(vocab, caption_map):
    '''
    Input:
        caption_map should be one of train_caption_id_2_caption, val_caption_id_2_caption, test_caption_id_2_caption
    Does:
        Replace infrequent words and out-of-vocab words with <UNK>
        prepend <SOS> and append <EOS> to all captions in |caption_map|
    '''
    logging.info("Start: Replace infrequent words with <UNK>, prepend <SOS>, append <EOS>.")
    vocab_set = set(vocab)
    total_count = 0 # Total number of words
    unk_count = 0   # Number of words set to <UNK>
    for id, caption in caption_map.items():
        new_caption = ["<SOS>"]
        for i, w in enumerate(caption.split(" ")):
            total_count += 1
            if w not in vocab_set:
                unk_count += 1
                new_caption.append("<UNK>")
            else:
                new_caption.append(w)
        new_caption.append("<EOS>")
        caption_map[id] = " ".join(new_caption)
    logging.info("Finished: Replaced {} of the {} words with <UNK>.".format(unk_count, total_count))


def generate_length_map(train_caption_id_2_caption):
    '''
    Returns:
        length_2_caption_id: {caption_length: [list of caption ids]}
        length_2_num_captions: {caption_length: count}
    '''
    logging.info("Start: Generate mapping from caption length to list of caption ids for training set.")
    length_2_caption_id = {}
    length_2_num_captions = {}
    for id, caption in train_caption_id_2_caption.items():
        length = len(caption.split(" "))
        if length not in length_2_caption_id:
            length_2_caption_id[length] = [id]
        else:
            length_2_caption_id[length].append(id)
        length_2_num_captions[length] = length_2_num_captions.get(length, 0) + 1

    for l in sorted(length_2_num_captions.keys()):
        # This is also a sanity check to see if we correctly truncated the captions
        logging.info("Length {}: {}".format(l, length_2_num_captions[l]))

    logging.info("Finished: Generate mapping from caption length to list of caption ids for training set.")
    return length_2_caption_id, length_2_num_captions

def shuffle_map(caption_map):
    '''
    Input:
        caption_map should be one of train_caption_id_2_caption, val_caption_id_2_caption, test_caption_id_2_caption
    Does:
        Shuffle the keys in the maps so that we can better iterate through them
    Returns:
        The shuffled map
    '''
    keys = list(caption_map.keys())
    random.shuffle(keys)
    new_map = {k:caption_map[k] for k in keys}

    assert set(new_map.keys()) == set(caption_map.keys())
    for k, v in new_map.items():
        assert v == caption_map[k]

    return new_map


def main(params):
    logging.info("Start preprocessing!")
    for key, val in params.items():
        logging.info("{}: {}".format(key, val))

    captions_dir = 'coco/annotations'
    train_ann_path = os.path.join(captions_dir, 'captions_train2014.json')
    val_ann_path = os.path.join(captions_dir, 'captions_val2014.json')

    # Generate short json summary file
    generate_short_json(captions_dir)

    # Setup Coco API
    coco_train = COCO(train_ann_path)
    coco_val = COCO(val_ann_path)
    coco_val_imgid = coco_val.getImgIds()

    # Perform train/val/test split. Store img_id of training/val/test images
    train_img_ids_1 = coco_train.getImgIds()
    val_img_ids, test_img_ids, train_img_ids_2 = split_data(coco_val_imgid, params)

    # Generate useful maps
    val_img_set, test_img_set = set(val_img_ids), set(test_img_ids)
    caption_id_2_img_id, train_caption_id_2_caption, val_caption_id_2_caption, \
        test_caption_id_2_caption = generate_id_maps(coco_train, coco_val, val_img_set, test_img_set)

    # Generate the trimmed GloVe matrix with only frequent words
    glove_path = 'glove.6B.300d.txt'
    trimmed_path = 'glove.6B.300d.trimmed.txt'
    vocab = generate_vocab(glove_path, trimmed_path, train_caption_id_2_caption)

    # Modify all the captions (<UNK>, <SOS>, <EOS>)
    modify_captions(vocab, train_caption_id_2_caption)
    modify_captions(vocab, val_caption_id_2_caption)
    modify_captions(vocab, test_caption_id_2_caption)

    # Generate mapping from caption length to list of caption ids -> better training efficiency
    length_2_caption_id, length_2_num_captions = generate_length_map(train_caption_id_2_caption)

    # Shuffle the three caption_id_2_caption maps (otherwise the keys are in order)
    train_caption_id_2_caption = shuffle_map(train_caption_id_2_caption)
    val_caption_id_2_caption = shuffle_map(val_caption_id_2_caption)
    test_caption_id_2_caption = shuffle_map(test_caption_id_2_caption)

    # Dump all the mappings, can be later loaded by cPickle.load(open(file_name, 'rb'))
    logging.info("Start: Save all the generated mappings to disk.")
    data_dir = "data/"
    cPickle.dump(caption_id_2_img_id, open(data_dir + "caption_id_2_img_id.p", "wb"))
    cPickle.dump(train_caption_id_2_caption, open(data_dir + "train_caption_id_2_caption.p", "wb"))
    cPickle.dump(val_caption_id_2_caption, open(data_dir + "val_caption_id_2_caption.p", "wb"))
    cPickle.dump(test_caption_id_2_caption, open(data_dir + "test_caption_id_2_caption.p", "wb"))
    cPickle.dump(length_2_caption_id, open(data_dir + "length_2_caption_id.p", "wb"))
    cPickle.dump(length_2_num_captions, open(data_dir + "length_2_num_captions.p", "wb"))

    logging.info("Successfully finished caption preprocessing! Bye!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', default=18, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_val', default=5000, type=int, help='number of validation images')
    parser.add_argument('--num_test', default=5000, type=int, help='number of test images')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler("prepro_log.txt")
    logging.getLogger().addHandler(file_handler)

    main(params)
