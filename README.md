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

Options: --primary_metric, --num_epochs, --learning_rate, --dropout, --batch_size, --hidden_size, --beam_width, --special_token, --print_every, --save_every, --eval_every

# Evaluate
python main.py --mode=eval --experiment_name=baseline --ckpt_load_dir=./experiments/baseline/best_checkpoint --data_source=ram/ssd

Note: The optimal beam width can be tuned by only running in eval mode for multiple times (although using the best beam width for early stopping during training might give slightly better performance)

# Baseline Experiments
For the baseline model, lr = 2e-4, p_drop = 0.2, hidden_size = 512 is a good set of hyperparameters. It was also found that adding bias in the final projection layer doesn't affect model performance at all. Beam width 3 outperforms all other values - 3 is also what Google's Show-and-Tell and Salesforce's Knowing-When-to-Look used. These hyperparameters are now set to be default in main.py

As of 05-17, the best baseline model achieves CIDEr 93.9, Bleu-4 30.5, METEOR 24.7, ROUGE 52.4

# Attention Model Experiments
The attention model using the custom BasicAttentionLayer with tri-linear similarity function (implemented and trained on 05-21) achieves CIDEr 96.7, Bleu-4 31.0, METEOR 25.2, ROUGE 52.9. The model was not tuned at all, so it can definitely be improved by careful tuning (is it worth it though?)

Next Step: Experiment with Salesforce's "Knowing When to Look" model. Need to create a custom SentinelLSTM cell (subclass tensorflow.python.ops.rnn_cell_impl.LayerRNNCell), and create a new SentinelAttention layer (subclass tensorflow.python.layers.base.layer)
