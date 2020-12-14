# An Unsupervised Joint System for Text Generation from Knowledge Graphs and Semantic Parsing
This repository contains the code for the [EMNLP 2020 long paper](https://www.aclweb.org/anthology/2020.emnlp-main.577) "An Unsupervised Joint System for Text Generation from Knowledge Graphs and Semantic Parsing".

If this code is useful for your work, please consider citing:
```
@inproceedings{schmitt-etal-2020-unsupervised,
    title = "An Unsupervised Joint System for Text Generation from Knowledge Graphs and Semantic Parsing",
    author = {Schmitt, Martin  and
      Sharifzadeh, Sahand  and
      Tresp, Volker  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.577",
    doi = "10.18653/v1/2020.emnlp-main.577",
    pages = "7117--7130"
}
```

# Software dependencies
## Creating an environment
Main dependency is AllenNLP. At the moment our compatibility is with the older version of 0.9.0.
You can install all the requirements from the `requirements.txt`. 
If you are using conda, create a new environment and activate it:

```
conda create --name kgtxt python=3.6
conda activate kgtxt
```

If you are **not** using conda, a virtual environment can also be created and activated like this:
```
virtualenv --python=3.6 env
source env/bin/activate
```

Then go to the project folder and install the requirements:

`pip install -r requirements.txt`

## "argmax_cuda" bug fix
We have noticed that the software environment that is necessary to recreate our experiments
causes an error during inference, namely when the argmax is to be computed over a bool tensor on the GPU.

Fortunately, this error is easy to fix.
If you installed AllenNLP in a virtual environment stored in the directory `env` as described above, then you have to change line 272 in the file `env/lib/python3.6/site-packages/allennlp/models/encoder_decoders/copynet_seq2seq.py` from
```
first_match = ((matches.cumsum(-1) == 1) * matches).argmax(-1)
```
to
```
first_match = ((matches.cumsum(-1) == 1) * matches).float().argmax(-1)
```

If you installed the software requirements according to our `requirements.txt`,
we recommend making this change before attempting to run any experiment.

# Datasets
## Visual Genome
The VG benchmark can be downloaded from [here](http://cistern.cis.lmu.de/unsupervised-graph-text-conversion).
Please read the dataset README for more details on the data format.

VG is licensed under a
[Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

## WebNLG v2.1
We have included the dataset in the 
dataset folder. The files were originally downloaded from [here](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v2.1/json).

The WebNLG data are licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

In case you want to retrace our preprocessing, we ran `LC_ALL=C.UTF-8 LANG=C.UTF-8 python src/data/format_webnlg.py input_file.json output_file.tsv` to preprocess each json file and convert it to a tsv file.
(`input_file.json` and `output_file.tsv` are to be replaced with the corresponding files in the dataset folder, e.g., `test.tsv`, `train.tsv`, `val.tsv`, etc.)

Depending on your system configuation, you might not need to explicitly enforce `LC_ALL=C.UTF-8 LANG=C.UTF-8`.

# Reproducing experiments
## Creating vocabularies
Before any model can be trained, the vocabulary has to be created with the script `src/data/make_vocab.py`. 
For WebNLG, e.g., create the folders `mkdir -p vocabularies/webNLG/` if they are not present. 
Then you can run: `python -m src.data.make_vocab data/webNLG/train.tsv data/webNLG/val-pairs.tsv vocabularies/webNLG`.
The process is analogous for VG with `vocabularies/vg`.

## Unsupervised training
### General instructions
An unsupervised model is trained with the following template:
```
CUDA_VISIBLE_DEVICES=<gpu_num> src/unsupervised.sh <model_dir> experiments/<path_to_experiment_dir>/unsupervised-lm.json experiments/<path_to_experiment_dir>/unsupervised-iter1.json <path_to_tsv_train_file> experiments/<backtranslation_config> <num_epochs> <vocabulary_dir>
```
where
- `<gpu_num>` is the id of the GPU you want to use for training (nonnegative integer).
- `<model_dir>` is the directory where intermediate model weights and backtranslated data will be stored. As the backtranslations from iteration i will be used as input in iteration i+1, this `<model_dir>` has to be the same as specified in the `unsupervised-iter1.json` to be used. **You have to adapt that file or create the default directory hierarchy in `models` before running the command.**
- `<path_to_experiment_dir>` is the location of the json configuration files to be used in the particular experiment. Each subdirectory of `experiments` contains one json file for the language modeling epoch (ending in `-lm.json`) and one json file that will be used for all other epochs (ending in `iter1.json`). Both have to be specified here.
- `<path_to_tsv_train_file>` is the training file as obtained from the data preprocessing scripts.
- `<backtranslation_config>` is the configuration file used for backtranslation, i.e., `unsupervised-bt-webNLG.jsonnet` for webNLG or `unsupervised-bt-VG.jsonnet` for VG.
- `<num_epochs>` specified the number of epochs/iterations for training. VG models were trained for 5 iterations and webNLG models for 30 iterations.
- `<vocabulary_dir>` is the directory where the vocabulary was stored with `make_vocab.py`; in the example above it is `vocabularies/webNLG`.

### Example
Let's say that you want to train webNLG with composed noise.

First, make sure that the 
parameters in `experiments/webNLG-composed/unsupervised-iter1.json` and `experiments/webNLG-composed/unsupervised-lm.json`
are set correctly. Specifially the `directory_path` of vocabularies, and `train_data_path` should point to the corresponding directories. Note that `train_data_path` contains a json dictionary (with escaped special characters) to include different training files for denoising loss and backtranslation loss.

Now Run:
```
CUDA_VISIBLE_DEVICES=0 LC_ALL=C.UTF-8 LANG=C.UTF-8 src/unsupervised.sh models/webNLG-composed experiments/webNLG-composed/unsupervised-lm.json experiments/webNLG-composed/unsupervised-iter1.json data/webNLG/train.tsv experiments/unsupervised-bt-webNLG.jsonnet 30 vocabularies/webNLG
```
(Note: Make sure that (1) you can run the script, e.g., run
 `chmod +x src/unsupervised.sh` and that (2) the model directory exist, e.g., run `mkdir -p models/webNLG-composed`.
)

## Supervised training
Supervised training uses the standard command from AllenNLP:
```
CUDA_VISIBLE_DEVICES=<gpu_num> allennlp train experiments/<path_to_config_file> -s <model_dir> --include-package src.features.copynet_shared_decoder --include-package src.models.copynet_shared_decoder
```
with
- `<gpu_num>` the id of the GPU to be used as above.
- `<path_to_config_file>` the location of the config file for supervised training. You will have to adapt `train_data_path`, `validation_data_path` and `directory_path` to the correct paths in your system.
- In the `<model_dir>` directory, all results such as metrics and model weights will be stored.

## Evaluation
Let's first create directories for predictions:

```
mkdir -p predictions/text2graph predictions/graph2text
```

The evaluation has two steps:
1. Generate the predictions
2. compare the predictions to the ground truth

For step 1, we can either batch-produce predictions from every iteration (with the script `src/produce_predictions.sh`) or produce predictions from a specific model (with `src/analysis/predict.py`).

Batch-predictions work like this:
```
CUDA_VISIBLE_DEVICES=<gpu_num> src/produce_predictions.sh <model_dir> experiments/unsupervised-bt-webNLG.jsonnet <dataset> <predictions_dir>
```
where `<dataset>` is one of the `-text2graph.tsv` or ``-graph2text.tsv` files for evaluation.

Predictions from a single model are obtained like this:
```
CUDA_VISIBLE_DEVICES=<gpu_num> python3 -m src.analysis.predict <config_file> path/to/model/best.th <dataset> hypo.txt ref.txt <batch_size>
```
where `<config_file>` is `unsupervised-bt-webNLG.jsonnet` for webNLG or `unsupervised-bt-VG.jsonnet` for VG
and `<batch_size>` is any positive integer (higher values need more memory but usually speed up decoding).


For step 2, we can batch-evaluate a directory of predictions with `evaluate_dir.sh` like this:

```
src/evaluate_dir.sh <predictions_dir> <eval_command>
```

where `eval_command` can either be

`'python3 -m src.analysis.multi-f1'`

for text-to-graph or

`'python3 -m src.bleu.multi-bleu'`

for graph-to-text.

Or we directly apply one of the two evaluation commands like this:
```
python3 -m src.analysis.multi-f1 hypo.txt ref.txt
```

Note: For evaluation on val100, you can add the flag --val100 to either of the two evaluation commands, e.g., `python3 -m src.bleu.multi-bleu --val100 hypo.txt ref.txt`.
