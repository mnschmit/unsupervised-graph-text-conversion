from typing import List, Tuple, FrozenSet
from argparse import ArgumentParser, Namespace
import random
import sys

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def preprocess_ssgp(fact_set: FrozenSet[Tuple[str, str, str]]) -> FrozenSet[Tuple[str, str, str]]:
    kept_facts = []
    for fact in fact_set:
        if not (fact[0].endswith("'") or fact[2].endswith("'")):
            kept_facts.append(fact)
    return frozenset(kept_facts)


def lemmatize_relations(fact_set: FrozenSet[Tuple[str, str, str]])\
        -> FrozenSet[Tuple[str, str, str]]:
    new_facts = []
    for fact in fact_set:
        new_facts.append(
            (fact[0], lemmatizer.lemmatize(fact[1], pos='v'), fact[2])
        )
    return frozenset(new_facts)


def generate_triple(elements: List[str], lower: bool = True) -> Tuple[str, str, str]:
    if elements:
        sbj = elements.pop(0)
    else:
        sbj = ''
    if elements:
        prd = elements.pop(0)
    else:
        prd = ''
    if elements:
        obj = elements.pop(0)
    else:
        obj = ''

    if lower:
        return sbj.lower(), prd.lower(), obj.lower()
    else:
        return sbj, prd, obj


def factseq2set(seq: str, eof=" @EOF@ ", sep=" @SEP@ ",
                lower=True) -> FrozenSet[Tuple[str, str, str]]:
    fact_strings: List[str] = seq.split(eof)
    facts: List[Tuple[str, str, str]] = [
        generate_triple(fact_string.split(sep))
        for fact_string in fact_strings
    ]
    return frozenset(facts)


def main(args: Namespace):
    num_correct_predicted = 0
    num_gt_facts = 0
    num_predicted_facts = 0

    with open(args.hypo_file) as hf, open(args.ref_file) as rf:
        data = list(zip(hf.readlines(), rf.readlines()))

    if args.val100:
        rand = random.Random(0)
        data = rand.sample(data, 100)

    for hline, rline in data:
        hypo_facts = factseq2set(hline.strip())
        refs = [
            factseq2set(ref_seq) for ref_seq in rline.strip().split('*#')
        ]
        num_gt_facts += max(len(r) for r in refs)
        num_predicted_facts += len(hypo_facts)

        if args.ssgp:
            hypo_facts = preprocess_ssgp(hypo_facts)
            refs = [lemmatize_relations(r) for r in refs]

        for hfact in hypo_facts:
            if any(hfact in r for r in refs):
                num_correct_predicted += 1

    precision = float(num_correct_predicted) / \
        float(num_predicted_facts + 1e-13)
    recall = float(num_correct_predicted) / float(num_gt_facts + 1e-13)
    f1 = 2. * ((precision * recall) / (precision + recall + 1e-13))

    print(
        "P: {:.1f} / R: {:.1f} / F1: {:.1f}".format(
            precision * 100, recall * 100, f1 * 100
        ), file=sys.stderr
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('hypo_file')
    parser.add_argument('ref_file')
    parser.add_argument('--val100', action='store_true')
    parser.add_argument('--ssgp', action='store_true')
    args = parser.parse_args()
    main(args)
