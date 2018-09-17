# Baseline (BiDAF) for Towards Exploiting Background Knowledge for Building Conversation Systems (EMNLP 2018)
We continue to use the [original implementation](https://github.com/allenai/bi-att-flow). We have only modified some of the data pre-processing files. The ROUGE and BLEU scripts have been taken from [this repository](https://github.com/google/seq2seq). For reproducibility of numbers, please use the parameters mentioned in this repository.

## Bi-directional Attention Flow for Machine Comprehension 
 

## 0. Requirements
#### General
- Python (verified on 3.5.2. Issues have been reported with Python 2!)
- unzip, wget (for running `download.sh` only)

#### Python Packages
- tensorflow (deep learning library, only works on r0.11)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```
After the download is complete, use the files from `mixed_short` or `mixed_long` from the data folder and copy them in to`$HOME/data/squad` 

Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training
The model has ~2.5M parameters.
The model was trained with NVidia Titan X (Pascal Architecture, 2016).
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
We train till 25k steps saving at every 5k steps

Before training, it is recommended to first try the following code to verify everything is okay and memory is sufficient:
```
python -m basic.cli --mode train --noload --debug
```

Then to fully train, run:
```
python -m basic.cli --mode train --noload
```

You can speed up the training process with optimization flags:
```
python -m basic.cli --mode train --noload --len_opt --cluster
```
You can still omit them, but training will be much slower.

Note that during the training, the EM and F1 scores from the occasional evaluation are not the same with the score from official squad evaluation script. 
We select the model based on best F1 score. (This has been automated)

## 3. Test
To test, run:
```
python -m basic.cli
```

Similarly to training, you can give the optimization flags to speed up test (5 minutes on dev data):
```
python -m basic.cli --len_opt --cluster
```

This command loads the most recently saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
To obtain the official number, use the official evaluator (i.e, the Squad script for F1) (copied in `squad` folder) and the output json file:

```
python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-####.json
```


## Multi-GPU Training & Testing
The model supports multi-GPU training.
They follow the parallelization paradigm described in [TensorFlow Tutorial][multi-gpu].
In short, if you want to use batch size of 60 (default) but if you have 3 GPUs with 4GB of RAM,
then you initialize each GPU with batch size of 20, and combine the gradients on CPU.
This can be easily done by running:
```
python -m basic.cli --mode train --noload --num_gpus 3 --batch_size 20
```

Similarly, you can speed up your testing by:
```
python -m basic.cli --num_gpus 3 --batch_size 20 
```
