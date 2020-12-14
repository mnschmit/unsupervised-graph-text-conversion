import click
import logging
from allennlp.data.vocabulary import Vocabulary
from ..features.copynet_shared_decoder import CopyNetSharedDecoderDatasetReader
import functools


@click.command()
@click.argument("train_file_path",
                type=click.Path(exists=True, dir_okay=False))
@click.argument("val_file_path",
                type=click.Path(exists=True, dir_okay=False))
@click.argument("vocab_dir",
                type=click.Path(exists=True, file_okay=False))
@click.option("--max-vocab-size",
              default=100000, type=click.INT, show_default=True)
@click.option("--min-frq", default=2, type=click.INT, show_default=True)
@click.option("--additional", multiple=True)
def main(train_file_path, val_file_path, vocab_dir, max_vocab_size, min_frq, additional):
    logger = logging.getLogger(__name__)

    reader = CopyNetSharedDecoderDatasetReader("tokens")

    logger.info("Reading train file")
    train = reader.read(train_file_path)
    logger.info("Reading val file")
    val = reader.read(val_file_path)

    added_data = []
    for data in additional:
        logger.info("Adding additional data from {}".format(data))
        added_data.append(reader.read(data))

    if added_data:
        added_data = functools.reduce(lambda a, b: a+b, added_data)

    logger.info("Building vocabulary")
    logger.info("Minimal token frequency: {}".format(min_frq))
    logger.info("Max vocab size: {}".format(max_vocab_size))
    vocab = Vocabulary.from_instances(
        train + val + added_data,
        min_count={'tokens': min_frq},
        max_vocab_size=max_vocab_size
    )
    vocab.add_token_to_namespace('@COPY@', namespace='tokens')
    vocab.add_token_to_namespace('@BLANKED@', namespace='tokens')
    vocab.save_to_files(vocab_dir)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
