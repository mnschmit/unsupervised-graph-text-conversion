import argparse
from tqdm import tqdm

from ..features.make_noise import graph2text, text2graph


def main(args: argparse.Namespace):
    with open(args.dataset) as f, open(args.hypo_file, 'w') as f_hypo,\
            open(args.ref_file, 'w') as f_ref:
        for line in tqdm(f):
            sample_id, src, trg_lang, trg, *rest = line.strip().split("\t")

            if trg_lang == "text":
                pred = graph2text(src)
            elif trg_lang == "graph":
                pred = text2graph(src)
            else:
                continue

            if args.add_sample_id:
                pred = sample_id + '\t' + pred
                trg = sample_id + '\t' + trg
            print(pred, file=f_hypo)
            print(trg, file=f_ref)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('hypo_file')
    p.add_argument('ref_file')
    p.add_argument('--add-sample-id', action='store_true')
    args = p.parse_args()

    main(args)
