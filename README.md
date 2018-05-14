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

# Train
python main.py --mode=train --experiment_name=baseline --data_source=ram/ssd

Options: --primary_metric, --num_epochs, --learning_rate, --dropout, --batch_size, --hidden_size, --beam_width, --print_every, --save_every, --eval_every

# Evaluate
python main.py --mode=eval --experiment_name=baseline --ckpt_load_dir=./experiments/baseline/best_checkpoint --data_source=ram/ssd

Note: The optimal beam width can be tuned by only running in eval mode for multiple times
