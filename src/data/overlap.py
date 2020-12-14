#!/usr/bin/python3

from typing import List
from commandr import command, Run
import csv
import logging
from tqdm import tqdm
import re


@command
def purge(dataset_purge, dataset_keep, suffix="_purged", delimiter="\t"):
    logger = logging.getLogger(__name__)

    index = index_data(dataset_keep, delimiter)

    new_dataset_path = re.sub(r'(\.[a-z]+)$', suffix + r'\1', dataset_purge)
    assert new_dataset_path != dataset_purge

    num_kept = 0
    num_total = 0
    logger.info("Purging {}".format(dataset_purge))
    with open(new_dataset_path, 'w') as fout, open(dataset_purge) as f:
        r = csv.reader(f, delimiter=delimiter)
        w = csv.writer(fout, delimiter=delimiter)
        for row in tqdm(r):
            inst = inst_repr(row)
            if inst not in index:
                w.writerow(row)
                num_kept += 1
            num_total += 1

    print(
        "Kept {} rows ({:.2f}%), deleted {} rows.".format(
            num_kept, num_kept/num_total*100, num_total - num_kept
        )
    )


def index_data(d, delimiter):
    logger = logging.getLogger(__name__)
    logger.info("Indexing {}".format(d))
    with open(d) as f:
        r = csv.reader(f, delimiter=delimiter)
        index = set()
        for row in tqdm(r):
            inst = inst_repr(row)
            index.add(inst)
    return index


def find_overlap(data1, data2, delimiter):
    logger = logging.getLogger(__name__)

    index = index_data(data2, delimiter)

    logger.info("Looking for duplicates")
    with open(data1) as f:
        r = csv.reader(f, delimiter=delimiter)
        for row in tqdm(r):
            candidate = inst_repr(row)
            if candidate in index:
                yield candidate


def inst_repr(row: List[str]):
    try:
        if row[2] == "graph":
            graph_idx = 3
            text_idx = 1
        else:
            graph_idx = 1
            text_idx = 3

        return (
            row[text_idx],
            tuple(sorted(row[graph_idx].split(" @EOF@ ")))
        )
    except IndexError as e:
        print("IndexError:", row)
        exit(-1)


@command
def count(dataset1, dataset2, do_print=False, delimiter="\t"):
    num_overlap = 0
    for overlap_inst in find_overlap(dataset1, dataset2, delimiter):
        if do_print:
            print(overlap_inst)
        num_overlap += 1

    print("Number of overlapping instances: {}".format(num_overlap))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    Run()
