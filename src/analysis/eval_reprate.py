import argparse
import nltk
from collections import Counter
from tqdm import tqdm


def compute_repetition_rate(s: str, max_n=4):
    res = []
    for n in range(1, max_n+1):
        ngrams = nltk.ngrams(nltk.word_tokenize(s), n)
        counts = Counter(ngrams)

        try:
            res.append(
                sum([1.0 for v in counts.values() if v > 1]) / len(counts))
        except ZeroDivisionError:
            res.append(0.0)

    return tuple(res)


def main(args: argparse.Namespace):
    cum_sum = [0]*4
    num_lines = 0
    with open(args.hypo_file) as f:
        for line in tqdm(f):
            sent = line.strip()
            rep_rates = compute_repetition_rate(sent)
            for i, r in enumerate(rep_rates):
                cum_sum[i] += r
            num_lines += 1
    for n, s in enumerate(cum_sum, 1):
        print('Repetitive {}-grams: {:.1f}%'.format(n, s*100 / num_lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('hypo_file')
    args = p.parse_args()

    main(args)
