from sklearn.model_selection import train_test_split
from os import path
import click
import logging
from collections import defaultdict


def len_class(l, steps=(10, 15, 20, 25, 30)):
    under_bin = 0
    while under_bin < len(steps) and l > steps[under_bin]:
        under_bin += 1

    if under_bin < len(steps):
        return "<= {}".format(steps[under_bin])
    else:
        return "> {}".format(steps[-1])


def serialize_data(data, image_ids, file_path, logger):
    logger.info("Writing {}".format(file_path))
    with open(file_path, 'w') as fout:
        for img_id in image_ids:
            for source, target_lang, target in data[img_id]:
                print(img_id, source, target_lang, target, sep="\t", file=fout)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False))
def main(input_filepath, output_dir):
    """ Turns extracted data from visual genome (data/interim) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading data.")
    data = defaultdict(list)
    image_ids = []
    sum_length = defaultdict(int)
    img_counter = defaultdict(int)
    with open(input_filepath) as f:
        for line in f:
            sample_id, image_id, source, target_lang, target = line.strip().split("\t")
            data[image_id].append((source, target_lang, target))

            if not image_ids or image_ids[-1] != image_id:
                image_ids.append(image_id)

            sum_length[image_id] += len(source.split()) + len(target.split())
            img_counter[image_id] += 1

    length_classes = [
        len_class(sum_length[image_id] / img_counter[image_id])
        for image_id in image_ids
    ]

    logger.info('Randomly splitting the images into train/val/test.')
    images_train, images_val_test, lc_train, lc_val_test = train_test_split(
        image_ids, length_classes, test_size=.2, random_state=42635,
        stratify=length_classes
    )
    images_val, images_test = train_test_split(
        images_val_test, test_size=.5, random_state=42635,
        stratify=lc_val_test
    )

    serialize_data(data, images_train, path.join(
        output_dir, "train.tsv"), logger)
    serialize_data(data, images_val, path.join(output_dir, "val.tsv"), logger)
    serialize_data(data, images_test, path.join(
        output_dir, "test.tsv"), logger)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
