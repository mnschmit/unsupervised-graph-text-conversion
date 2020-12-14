#!/usr/bin/python3

import click
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def len_class(l, steps=(10,20,30,40,50,60)):
    under_bin = 0
    while under_bin < len(steps) and l > steps[under_bin]:
        under_bin += 1

    if under_bin < len(steps):
        return "<= {}".format(steps[under_bin])
    else:
        return "> {}".format(steps[-1])


@click.command()
@click.argument("data_file", type=click.Path(dir_okay=False))
def main(data_file):
    # steps = (200,400,600,800,1000,1200)
    steps = (10,15,20,25,30)
    
    length_of_img = defaultdict(int)
    img_counter = defaultdict(int)
    with open(data_file) as f:
        for line in tqdm(f):
            sample_id, image_id, str1, str2 = line.split("\t")
            length_of_img[image_id] += len(str1.split()) + len(str2.split())
            img_counter[image_id] += 1

    length_count = defaultdict(int)
    for image_id, l in length_of_img.items():
        length_count[len_class(l / img_counter[image_id], steps=steps)] += 1

    x = list(sorted(length_count.keys()))
    heights = [length_count[l] for l in x]
    plt.bar(x, heights)
    plt.show()
            
if __name__ == "__main__":
    main()
