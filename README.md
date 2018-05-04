# CS231N_Project
Image Captioning on Microsoft Coco Dataset

# Getting Started
1. Clone this repository
2. Download the Microsoft Coco dataset from http://cocodataset.org/#download. We need 2014 Train images, 2014 Val Images, and 2014 Train/Val annotations. Move the zip files "train2014.zip", "val2014.zip", and "annotations_trainval2014.zip" to the coco/ folder, and unzip them.
3. Download the glove.6B.300d word vectors from http://nlp.stanford.edu/data/glove.6B.zip. Place the "glove.6B.300d.txt" file in the top-level folder.
4. Run the "get_started.sh" script. This will re-organize the data files, and build the Coco Python API.
5. If interested, you can check out "explore_data.ipynb" to see what the official json files look like.

# Perform train/val/test split and preprocess all the captions
1. Run "python prepro.py". It should take less than two minutes. The script is tested on Python 3.6. It depends on numpy and nltk. After running the script, there will be several data files generated in the data/ folder, and a trimmed version of glove.6B.300d will be saved to the top-level folder. For details about the generated files, see the comments in "prepro.py"
2. Start Jupyter Notebook and open "prepro_unit_test.ipynb", re-run all the cells in the notebook to make sure the preprocessing was successful.

# Next Steps
Preprocess all the images to generate image features, and save all the features to a Pickle file in the "data/" directory. The saved data should look like {image_id (int): image_features (list or numpy array)}. Specifically, the image preprocessing script should scan the coco/images/train and coco/images/val folders, and preprocess them one by one. The "image_id" is simply the last few digits of the image file names, so it should be easy to extract them when scanning the folders using the os package.
