from __future__ import print_function

from .bleu import Bleu
import argparse
import random
import sys


def make_new(*dicts, **kwargs):
    res = []
    for d in dicts:
        tmp = {}
        for k in sorted(d.keys()):
            if 'keys' not in kwargs or k in kwargs['keys']:
                tmp[k] = d[k]
        res.append(tmp)
    return res


def collect_dict(filename):
    res = {}
    with open(filename) as f:
        for sample_id, line in enumerate(f):
            text = line.strip()
            res[sample_id] = text.split('*#')

    return res


def compute_bleu_score(ref_dict, hypo_dict, random100=False):
    if random100:
        rng = random.Random(0)
        sampled_keys = rng.sample(set(ref_dict.keys()), 100)
        ref_dict, hypo_dict = make_new(ref_dict, hypo_dict, keys=sampled_keys)

    b = Bleu()
    score, scores = b.compute_score(ref_dict, hypo_dict)
    return score[-1]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('hypo_file')
    p.add_argument('ref_file')
    p.add_argument('--val100', action='store_true')
    args = p.parse_args()

    hypo_dict = collect_dict(args.hypo_file)
    ref_dict = collect_dict(args.ref_file)

    score = compute_bleu_score(ref_dict, hypo_dict, random100=args.val100)
    print("BLEU:", score, file=sys.stderr)
