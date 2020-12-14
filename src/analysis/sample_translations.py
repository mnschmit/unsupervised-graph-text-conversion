import click
import logging

import torch
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.models.model import Model
from allennlp.predictors.seq2seq import Seq2SeqPredictor

from ..features.copynet_shared_decoder import *
from ..models.copynet_shared_decoder import *
from .f1_score import *

import random


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_th", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.argument("seed", type=click.INT)
def main(config: str, model_th: str, dataset: str, seed: int):
    logger = logging.getLogger(__name__)

    logger.info("Loading model and data")
    params = Params.from_file(config)

    vocab_params = params.pop("vocabulary")
    vocab = Vocabulary.from_params(vocab_params)

    reader_params = params.pop("dataset_reader")
    reader_name = reader_params.pop("type")
    reader_params["lazy"] = True  # make sure we do not load the entire dataset
    reader = DatasetReader.by_name(
        reader_name
    ).from_params(reader_params)

    data = reader.read(dataset)

    iterator = BasicIterator(batch_size=10)
    iterator.index_with(vocab)

    batches = iterator._create_batches(data, shuffle=False)

    model_params = params.pop("model")
    model_name = model_params.pop("type")
    model = Model.by_name(model_name).from_params(model_params, vocab=vocab)
    # model.cuda(cuda_device)

    with open(model_th, 'rb') as f:
        model.load_state_dict(torch.load(f))

    predictor = Seq2SeqPredictor(model, reader)
    model.eval()

    logger.info("Generating predictions")

    random.seed(seed)
    samples = []
    for b in batches:
        samples.append(b)
        if random.random() > 0.6:
            break

    sample = list(random.choice(samples))
    pred = predictor.predict_batch_instance(sample)

    for inst, p in zip(sample, pred):
        print()
        print(
            "SOURCE:",
            " ".join([
                t.text
                for t in inst["source_tokens"]
            ])
        )
        print(
            "GOLD:",
            " ".join([
                t.text
                for t in inst["target_tokens"]
            ])
        )
        print(
            "GEN:",
            p["predicted_tokens"]
        )


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.WARNING, format=log_fmt)

    main()
