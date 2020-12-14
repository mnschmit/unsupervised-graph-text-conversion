from typing import List, FrozenSet, Tuple
import argparse
from tqdm import tqdm
from .ie_eval import factseq2set

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


def read_graph_list(filename: str, ssgp: bool = False,
                    ref: bool = False) -> List[FrozenSet[Tuple[str, str, str]]]:
    res = []
    with open(filename) as f:
        for line in tqdm(f):
            fact_set = factseq2set(line.strip().split())
            if ssgp:
                fact_set = preprocess_ssgp(fact_set)
            if ref:
                fact_set = lemmatize_relations(fact_set)
            res.append(fact_set)
    return res


def main(args: argparse.Namespace):
    hypo = read_graph_list(args.hypo, ssgp=True)
    ref = read_graph_list(args.ref, ref=True)

    correct_retrieved = 0
    retrieved = 0
    correct = 0
    for h, r in tqdm(zip(hypo, ref)):
        correct_retrieved += len(h & r)
        retrieved += len(h)
        correct += len(r)

    precision = float(correct_retrieved) / float(retrieved)
    recall = float(correct_retrieved) / float(correct)
    f1_score = 2. * ((precision * recall) / (precision + recall + 1e-13))

    print('P {:.1f}'.format(precision*100))
    print('R {:.1f}'.format(recall*100))
    print('F1 {:.1f}'.format(f1_score*100))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('hypo')
    p.add_argument('ref')
    args = p.parse_args()

    main(args)
