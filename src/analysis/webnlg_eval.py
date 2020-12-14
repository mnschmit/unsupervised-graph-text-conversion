import argparse
import os
from tqdm import tqdm
import re
import spacy


def reformat(s: str):
    return re.sub(
        r'([-–])', ' \1 ',
        re.sub(
            r',', ' ,',
            re.sub(
                r'\.$', ' .',
                re.sub(r'\s+', ' ', s)
            )
        )
    )


def retokenize(tokenizer, s: str):
    return re.sub(r'\s+', ' ', ' '.join([t.text for t in tokenizer(s)]))


def seq_based(args):
    spacy_model = spacy.load('en_core_web_sm', disable=[
                             'vectors', 'textcat', 'tagger', 'parser', 'ner'])

    with open(args.hypo_file) as fh, open(args.ref_file) as fr,\
            open(os.path.join(args.out_dir, 'hypo.txt'), 'w') as fh_out,\
            open(os.path.join(args.out_dir, 'ref.txt'), 'w') as fr_out:
        for hline, rline in tqdm(zip(fh, fr)):
            _, htext, _, hgraph = hline.strip().split('\t')
            sample_id, rgraph, _, rtext = rline.strip().split('\t')
            print(sample_id, htext, file=fh_out, sep='\t')
            print(sample_id, retokenize(
                spacy_model, rtext), file=fr_out, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hypo_file')
    parser.add_argument('ref_file')
    parser.add_argument('out_dir')
    args = parser.parse_args()

    seq_based(args)
