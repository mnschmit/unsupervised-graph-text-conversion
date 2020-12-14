#!/usr/bin/python3

import json
from tqdm import tqdm
import nltk
import click
import logging
from os import path
from collections import OrderedDict


def format_triple(sbj, pred, obj):
    return (sbj + " @SEP@ " + pred + " @SEP@ " + obj).replace('"', '')


def generate_triple(edge, object_id2name):
    sbj = object_id2name[edge["subject_id"]]
    obj = object_id2name[edge["object_id"]]
    pred = edge["predicate"].lower()
    return format_triple(sbj, pred, obj)


def extract_attributes(attribute_file, logger):
    logger.info("Loading attribute json file.")
    with open(attribute_file) as f:
        attributes = json.load(f)

    attr = {}
    for image in attributes:
        attr_dict = {}
        for obj in image["attributes"]:
            if "attributes" in obj:
                attr_dict[obj["object_id"]] = obj["attributes"]
        attr[image["image_id"]] = attr_dict

    return attr


def extract_parallel_data(region_graph_file, attr, logger):
    logger.info("Loading region graph json file.")
    with open(region_graph_file) as f:
        region_graphs = json.load(f)

    logger.info("Extracting non-duplicate parallel data.")
    data = {}
    for image in tqdm(region_graphs):
        attributes = attr[image["image_id"]]
        for region in image["regions"]:
            object_id2name = {}
            triples4region = OrderedDict()
            for obj in region["objects"]:
                object_id2name[obj["object_id"]] = obj["name"]
                if obj["object_id"] in attributes:
                    for a in attributes[obj["object_id"]]:
                        triples4region[
                            format_triple(obj["name"], "has_attribute", a)
                        ] = None

            for edge in region["relationships"]:
                triples4region[
                    generate_triple(edge, object_id2name)
                ] = None

            if triples4region:
                new_entry = (
                    image["image_id"],
                    " ".join(
                        nltk.word_tokenize(
                            region["phrase"]
                        )
                    ),
                    " @EOF@ ".join(triples4region.keys())
                )
                if new_entry not in data:
                    data[new_entry] = len(data)

    return data


def serialize(data, file_path, logger):
    logger.info("Writing {}".format(file_path))
    with open(file_path, 'w') as fout:
        for entry in tqdm(sorted(data.keys(), key=data.__getitem__)):
            print(data[entry], *entry, sep="\t", file=fout)


@click.command()
@click.argument("region_graph_file",
                type=click.Path(exists=True, dir_okay=False))
@click.argument("attribute_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir",
                type=click.Path(exists=True, file_okay=False, writable=True))
def extract_s2s(region_graph_file, attribute_file, output_dir):
    logger = logging.getLogger(__name__)

    attr = extract_attributes(attribute_file, logger)
    data = extract_parallel_data(
        region_graph_file, attr, logger
    )

    serialize(data, path.join(output_dir, "seq2seq.tsv"), logger)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    extract_s2s()
