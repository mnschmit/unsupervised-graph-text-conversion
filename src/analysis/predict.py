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


def main(config: str, model_th: str, dataset: str, hypo_file: str, ref_file: str,
         batch_size: int, no_gpu: bool):
    logger = logging.getLogger(__name__)

    logger.info("Loading configuration parameters")
    params = Params.from_file(config)

    vocab_params = params.pop("vocabulary")
    vocab = Vocabulary.from_params(vocab_params)

    reader_params = params.pop("dataset_reader")
    reader_name = reader_params.pop("type")
    # reader_params["lazy"] = True  # make sure we do not load the entire dataset

    reader = DatasetReader.by_name(reader_name).from_params(reader_params)

    logger.info("Reading data from {}".format(dataset))
    data = reader.read(dataset)

    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)
    batches = iterator._create_batches(data, shuffle=False)

    logger.info("Loading model")
    model_params = params.pop("model")
    model_name = model_params.pop("type")
    model = Model.by_name(model_name).from_params(model_params, vocab=vocab)
    if not no_gpu:
        model.cuda(0)

    with open(model_th, 'rb') as f:
        if no_gpu:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(f)

    model.load_state_dict(state_dict)

    predictor = Seq2SeqPredictor(model, reader)
    model.eval()

    with open(hypo_file, 'w') as hf, open(ref_file, 'w') as rf:
        logger.info("Generating predictions")
        for sample in tqdm(batches):
            s = list(sample)
            pred = predictor.predict_batch_instance(s)

            for inst, p in zip(s, pred):
                print(
                    " ".join(p["predicted_tokens"][0]),
                    file=hf
                )
                print(
                    " ".join(t.text for t in inst["target_tokens"][1:-1]),
                    file=rf
                )


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.WARNING, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('model_th')
    parser.add_argument('dataset')
    parser.add_argument('hypo_file')
    parser.add_argument('ref_file')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    main(args.config, args.model_th, args.dataset, args.hypo_file,
         args.ref_file, args.batch_size, args.no_gpu)
