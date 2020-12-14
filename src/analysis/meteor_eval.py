import argparse
import os
from tqdm import tqdm
import re


def preprocess(s: str):
    return re.sub(r'\s+', ' ', s)


def dict_based(args):
    hypo = {}
    with open(args.hypo_file) as f:
        for line in tqdm(f, desc='load hypo file'):
            sid, text, lang_id, graph = line.strip().split('\t')
            hypo[preprocess(graph)] = preprocess(text)

    ref = {}
    with open(args.ref_file) as f:
        for line in tqdm(f, desc='load ref file'):
            try:
                img_id, graph, lang_id, text = line.strip().split('\t')
            except ValueError:
                print('ERROR: not enough columns')
                print(line)
                exit(1)

            ref[preprocess(graph)] = preprocess(text)

    with open(os.path.join(args.out_dir, 'hypo.txt'), 'w') as f_hypo,\
            open(os.path.join(args.out_dir, 'ref.txt'), 'w') as f_ref:
        for graph in tqdm(ref, desc='write eval files'):
            print(ref[graph], file=f_ref)
            print(hypo[graph], file=f_hypo)


def seq_based(args):
    with open(args.hypo_file) as fh, open(args.ref_file) as fr,\
            open(os.path.join(args.out_dir, 'hypo.txt'), 'w') as fh_out,\
            open(os.path.join(args.out_dir, 'ref.txt'), 'w') as fr_out:
        for hline, rline in tqdm(zip(fh, fr)):
            sid, htext, _, hgraph = hline.strip().split('\t')
            img_id, rgraph, _, rtext = rline.strip().split('\t')
            print(htext, file=fh_out)
            print(rtext, file=fr_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hypo_file')
    parser.add_argument('ref_file')
    parser.add_argument('out_dir')
    args = parser.parse_args()

    seq_based(args)
