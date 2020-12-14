#!/usr/bin/python3

from commandr import command, Run
import numpy as np
from tqdm import tqdm
import random
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def lexicals_from_graph(graph, lower_case=False, filter_stop=False):
    global stop_words

    facts = graph.split("@EOF@")
    lexicals = set([
        t.lower() if lower_case else t
        for f in facts
        for l in f.split("@SEP@")
        for t in l.strip().split()
        if not (filter_stop and t in stop_words)
    ])
    return lexicals


def lexicals_from_text(text, lower_case=False, filter_stop=False):
    global stop_words

    return set([
        t.lower() if lower_case else t
        for t in text.split()
        if not (filter_stop and t in stop_words)
    ])


def generate_lexicals(filename, with_line=False, lower_case=False, filter_stop=False):
    with open(filename) as f:
        for line in tqdm(f):
            sample_id, src, trg_lang, trg = line.strip().split("\t")

            if trg_lang == "text":
                graph = src
                text = trg
            else:
                graph = trg
                text = src

            gl = lexicals_from_graph(
                graph, lower_case=lower_case, filter_stop=filter_stop)
            tl = lexicals_from_text(
                text, lower_case=lower_case, filter_stop=filter_stop)

            if with_line:
                yield gl, tl, line
            else:
                yield gl, tl


@command
def examples(filename, min=0.0, max=1.0):
    candidates = []
    for graph_lexicals, text_lexicals, line in generate_lexicals(filename, with_line=True):
        relative_overlap = len(
            graph_lexicals & text_lexicals
        ) / len(text_lexicals)

        if min <= relative_overlap <= max:
            candidates.append(line)

    if not candidates:
        print("No candidates found!")
        return

    print("Found {} candidates.".format(len(candidates)))

    for line in random.choices(candidates, k=10):
        print(line)


@command
def content_selection_examples(filename, text_thr=0.8, graph_thr=0.2):
    candidates = []
    for gl, tl, line in generate_lexicals(filename, with_line=True):
        rel_ov_text = len(gl & tl) / len(tl)
        rel_ov_graph = len(gl & tl) / len(gl)

        if rel_ov_text > text_thr and rel_ov_graph < graph_thr:
            candidates.append(line)

    if not candidates:
        print("No candidates!")
        return

    print("Found {} candidates.".format(len(candidates)))

    for line in random.choices(candidates, k=min(len(candidates), 10)):
        print(line)


@command
def ie_examples(filename, text_thr=0.2, graph_thr=0.8):
    candidates = []
    for gl, tl, line in generate_lexicals(filename, with_line=True):
        rel_ov_text = len(gl & tl) / len(tl)
        rel_ov_graph = len(gl & tl) / len(gl)

        if rel_ov_text < text_thr and rel_ov_graph > graph_thr:
            candidates.append(line)

    if not candidates:
        print("No candidates!")
        return

    print("Found {} candidates.".format(len(candidates)))

    for line in random.choices(candidates, k=min(len(candidates), 10)):
        print(line)


@command
def zero(filename):
    zeros = []
    for graph_lexicals, text_lexicals, line in generate_lexicals(filename, with_line=True):
        overlap = graph_lexicals & text_lexicals
        if not overlap:
            zeros.append(line)

    for line in random.choices(zeros, k=10):
        print(line)


@command
def stats(filename, lower=False, stop=False):
    absolute_overlap_sizes = []
    relative_overlap_sizes = []
    for graph_lexicals, text_lexicals in generate_lexicals(
            filename, lower_case=lower, filter_stop=stop
    ):
        absolute_overlap_sizes.append(len(graph_lexicals & text_lexicals))
        relative_overlap_sizes.append(
            len(graph_lexicals & text_lexicals) / len(text_lexicals)
        )

    print(
        "Avg absolute overlap size: {:.1f}\nStdev: {:.1f}\nMax: {}\nMin: {}".format(
            np.mean(absolute_overlap_sizes), np.std(absolute_overlap_sizes),
            max(absolute_overlap_sizes), min(absolute_overlap_sizes)
        )
    )
    print(
        "Avg relative overlap size: {:.3f}\nStdev: {:.3f}\nMax: {}\nMin: {}".format(
            np.mean(relative_overlap_sizes), np.std(relative_overlap_sizes),
            max(relative_overlap_sizes), min(relative_overlap_sizes)
        )
    )
    print(
        "{} ({:.1f}%) samples with 100% overlap\n{} ({:.1f}%) samples with 0% overlap".format(
            relative_overlap_sizes.count(1.0),
            relative_overlap_sizes.count(
                1.0) / len(relative_overlap_sizes) * 100,
            absolute_overlap_sizes.count(0),
            absolute_overlap_sizes.count(0) / len(absolute_overlap_sizes) * 100
        )
    )


if __name__ == "__main__":
    Run()
