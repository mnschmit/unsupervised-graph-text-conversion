#!/usr/bin/python3

import click
from tqdm import tqdm
import numpy as np


@click.command()
@click.argument("data_file", type=click.File())
def main(data_file):
    column_lengths = []

    for line in tqdm(data_file):
        columns = line.strip().split("\t")

        if not column_lengths:
            for col in columns:
                column_lengths.append([])

        for i in range(len(columns)):
            length = len(columns[i].split())
            column_lengths[i].append(length)

    for i, col in enumerate(column_lengths, 1):
        print(
            "Column {}: Avg length: {:.1f}, Stdev: {:.1f}, Max: {}".format(
                i, np.mean(col), np.std(col), max(col)
            )
        )


if __name__ == "__main__":
    main()
