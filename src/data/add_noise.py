#!/usr/bin/python3

from commandr import command, Run
import random
import csv
from tqdm import tqdm

from ..features.make_noise import noise_fun

@command
def run(data_in, data_out, noises=["swap", "drop", "rule", "repeat"], seed=1337):
    random.seed(seed)

    with open(data_in) as f, open(data_out, 'w') as fout:
        r = csv.reader(f, delimiter="\t")
        w = csv.writer(fout, delimiter="\t")
        for row in tqdm(r):
            img_id, src, trg_lang, trg = row
            noise = random.choice(noises)
            noised_src = noise_fun[noise](src, trg_lang == "text")

            # (1) supervised
            w.writerow(row)
            # (2) reconstruction
            w.writerow([img_id, noised_src, flip_target_lang[trg_lang], src])
            # (3) noisy supervised
            w.writerow([img_id, noised_src, trg_lang, trg])


if __name__ == "__main__":
    Run()
