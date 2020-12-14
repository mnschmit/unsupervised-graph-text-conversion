import click
import logging
import csv
from tqdm import tqdm

import torch
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.models.model import Model
from allennlp.predictors.seq2seq import Seq2SeqPredictor

from ..features.bt_reader import *
from ..features.coco_reader import UnsupervisedBTReader


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_th", type=click.Path(exists=True, dir_okay=False))
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_file", type=click.File('w'))
def main(config: str, model_th: str, dataset: str, out_file):
    logger = logging.getLogger(__name__)

    logger.info("Loading model and data")
    params = Params.from_file(config)

    vocab_params = params.pop("vocabulary")
    vocab = Vocabulary.from_params(vocab_params)

    reader_params = Params({
        "source_token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "tokens"
            }
        },
        "target_namespace": "tokens"
    })
    # reader_name = reader_params.pop("type")
    # reader_params["lazy"] = True  # make sure we do not load the entire dataset

    reader = UnsupervisedBTReader.from_params(reader_params)

    logger.info("Reading data from {}".format(dataset))
    data = reader.read(dataset)

    iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)
    batches = iterator._create_batches(data, shuffle=False)

    logger.info("Loading model")
    model_params = params.pop("model")
    model_name = model_params.pop("type")
    model = Model.by_name(model_name).from_params(model_params, vocab=vocab)
    model.cuda(0)

    with open(model_th, 'rb') as f:
        model.load_state_dict(torch.load(f))

    predictor = Seq2SeqPredictor(model, reader)
    model.eval()

    line_id = 0
    writer = csv.writer(out_file, delimiter="\t")
    logger.info("Generating predictions")
    for sample in tqdm(batches):
        s = list(sample)
        pred = predictor.predict_batch_instance(s)

        for inst, p in zip(s, pred):
            writer.writerow((
                line_id,
                " ".join((t.text for t in inst["source_tokens"][1:-1])),
                " ".join(p["predicted_tokens"][0])
            ))
            line_id += 1


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.WARNING, format=log_fmt)

    main()
