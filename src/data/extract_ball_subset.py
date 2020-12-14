#!/usr/bin/python3

import click
import logging


@click.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.argument("ball_file", type=click.Path(dir_okay=False))
def main(dataset, ball_file):
    logger = logging.getLogger(__name__)

    logger.info("Collecting all ball images.")
    ball_imgs = set()
    with open(dataset) as f:
        for line in f:
            img_id, source, target_lang, target = line.strip().split("\t")

            graph = target if target_lang == "graph" else source
            is_included = any([
                w.endswith("ball")
                for w in graph.split()
            ])
            if is_included:
                ball_imgs.add(img_id)

    logger.info("Writing out ball regions.")
    with open(dataset) as f, open(ball_file, 'w') as fout:
        for line in f:
            if line.split("\t")[0] in ball_imgs:
                print(line.strip(), file=fout)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
