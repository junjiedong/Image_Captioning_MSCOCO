"""
This files extract image features using Inception ResNet V2 model

Need file './coco/coco_raw.json', which is generated from prepro.py

Store image id to image features in './data/img_features.hdf5'. 

Only RGB image features are extracted. Other image raw data are stored in './data/other_img.p'

"""

import os
import _pickle as cPickle
import logging
import json
import numpy as np
import h5py
import PIL
from PIL import Image

import tensorflow as tf
from tensorflow.python.keras.applications import InceptionResNetV2
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input

def main():
	logging.info("Start preprocessing Images...")
	model = InceptionResNetV2(include_top=False, weights='imagenet')
	logging.info("Finish loading InceptionResNetV2 Model...")
	
	raw = json.load(open('data/coco_raw.json', 'r'))
	# img_features_map:  {image_id : image_features (height=8, width=8, channels=1536)}
	# used hdf5 to store the map
	img_features_map = h5py.File('data/img_features.hdf5','w')
	# other image ids are stored in a list
	other_img = []
	for i,img_raw in enumerate(raw):
		img = Image.open('coco/images/' + img_raw['file_path'])
		# reshape to 299*299 with high-quality downsampling filter
		img_resized = img.resize((299,299),PIL.Image.ANTIALIAS)
		img_resized = np.asarray(img_resized, dtype=np.uint8)
		img_id = img_raw['id']
		# transform image for preprocessing
		cnn_input = img_resized.copy()
		cnn_input = cnn_input.astype(np.float32)
		cnn_input = np.expand_dims(cnn_input,axis=0)
		# preprocess image, only use data with RGB 3 layers
		if cnn_input.shape==(1,299,299,3):
			cnn_input = preprocess_input(cnn_input)
			# extract features and stores the result
			pred = model.predict(cnn_input)
			img_features = np.squeeze(pred,axis=0)
			# reshape 8*8*1536 to 64*1536 
			assert img_features.shape == (8,8,1536)
			img_features = np.reshape(img_features,(64,1536))
			# store hdf5 file: image_features can be retrieved by using img_features_map[str(image_id)].value
			img_features_map.create_dataset(str(img_id),data=img_features)
		# store img_id for images with only one layer 
		else:
			other_img.append(img_raw)

		if i%1000==0:
			logging.info("Finished extracting {} images".format(i))

	img_features_map.close()
	cPickle.dump(other_img, open("data/other_img.p", "wb"))
	logging.info('Finished extracting all image features!')

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	main()