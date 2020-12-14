import argparse
from tqdm import tqdm
import logging


def canonical_graph_view(graph_ser: str, fact_sep=" @EOF@ ") -> str:
    return fact_sep.join(sorted(graph_ser.split(fact_sep)))


def main(args):
    data = {}
    samples = []
    logger = logging.getLogger(__name__)

    logger.info('Reading in dataset')
    with open(args.dataset) as f:
        for line in tqdm(f):
            img_id, task_input, task_id, reference = line.strip().split('\t')

            if args.canonical_graphs and task_id == 'text':
                task_input = canonical_graph_view(task_input)

            if task_input in data:
                data[task_input].append((img_id, task_id, reference))
            else:
                samples.append(task_input)
                data[task_input] = [(img_id, task_id, reference)]

    logger.info('Unifying data')
    with open(args.unified_data, 'w') as f:
        for sample_id, task_input in tqdm(enumerate(samples)):
            img_ids, task_ids, references = zip(*data[task_input])
            img_ids = set(img_ids)
            assert len(set(task_ids)) == 1
            print(sample_id, task_input,
                  task_ids[0], '*#'.join(references), str(img_ids),
                  sep='\t', file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('unified_data')
    parser.add_argument('--canonical-graphs', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
