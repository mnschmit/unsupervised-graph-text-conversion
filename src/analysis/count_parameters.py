import argparse
import logging
from tqdm import tqdm

import torch
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.models.model import Model
from allennlp.predictors.seq2seq import Seq2SeqPredictor

from ..features.copynet_shared_decoder import *
from ..models.copynet_shared_decoder import *


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config: str):
    logger = logging.getLogger(__name__)

    logger.info("Loading model and data")
    params = Params.from_file(config)

    vocab_params = params.pop("vocabulary")
    vocab = Vocabulary.from_params(vocab_params)

    logger.info("Loading model")
    model_params = params.pop("model")
    model_name = model_params.pop("type")
    model = Model.by_name(model_name).from_params(model_params, vocab=vocab)

    print("Number of parameters:", count_parameters(model))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    main(args.config)
