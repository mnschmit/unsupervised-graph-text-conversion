from typing import List

import argparse
from tqdm import tqdm

from ..features.make_noise import text2graph, graph2text
from .ie_eval import factseq2set
# from .ssgp_eval import lemmatize_relations
from nltk.translate.bleu_score import corpus_bleu


parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--sample", action='store_true')
args = parser.parse_args()

correct_retrieved = 0
retrieved = 0
correct = 0
predicted_texts: List[str] = []
reference_texts: List[List[str]] = []


with open(args.dataset) as f:
    for line_no, line in enumerate(tqdm(f)):
        img_id, src, trg_lang, trg = line.strip().split("\t")

        if trg_lang == "graph":
            pred = text2graph(src)
            # lemmatize_relations
            pred_facts = factseq2set(pred.split())
            true_facts = factseq2set(trg.split())

            if args.sample:
                print("HYPO:", pred_facts)
                print("REF:", true_facts)
                if line_no > 10:
                    exit(0)

            correct_retrieved += len(pred_facts & true_facts)
            retrieved += len(pred_facts)
            correct += len(true_facts)
        elif trg_lang == "text":
            pred = graph2text(src)
            predicted_texts.append(pred)
            reference_texts.append([trg])
        else:
            RuntimeError("impossible to get here")

precision = float(correct_retrieved) / float(retrieved)
recall = float(correct_retrieved) / float(correct)
f1_score = 2. * ((precision * recall) / (precision + recall + 1e-13))

print(
    "Text2Graph: {:.3f} P / {:.3f} R / {:.3f} F1".format(
        precision, recall, f1_score)
)

if reference_texts:
    print(
        "Graph2Text: {:.3f} BLEU".format(
            corpus_bleu(reference_texts, predicted_texts))
    )
