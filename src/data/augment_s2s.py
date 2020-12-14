import click
import logging
from tqdm import tqdm

@click.command()
@click.argument('input_file', type=click.File())
@click.argument('output_file', type=click.File(mode='w'))
def main(input_file, output_file):
    """ Augments extracted data from visual genome such that
    we have pairs for both directions (text <-> graph) and annotations.
    """
    logger = logging.getLogger(__name__)

    logger.info("Adding pairs of opposite direction and annotations")
    for line in tqdm(input_file):
        sample_id, image_id, descr, graph = line.strip().split("\t")

        print(
            sample_id, image_id, descr, "graph", graph,
            sep="\t", file=output_file
        )
        print(
            sample_id, image_id, graph, "text", descr,
            sep="\t", file=output_file
        )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
