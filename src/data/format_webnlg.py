from typing import Sequence, Tuple
import argparse
import json
from tqdm import tqdm


def preprocess_entity(ent: str, token_sep=' ') -> str:
    return ent.replace('_', token_sep).replace('"', '')


def preprocess_relation(rel: str, token_sep=' ') -> str:
    res_buffer = []
    mid_word = False
    for char in rel:
        if char == '_':
            res_buffer.append(token_sep)
            mid_word = False
        elif char.isupper() and mid_word:
            res_buffer.append(token_sep)
            res_buffer.append(char.lower())
            mid_word = False
        else:
            res_buffer.append(char)
            mid_word = char.islower()
    return ''.join(res_buffer)


def format_graph(triples: Sequence[Tuple[str, str, str]]) -> str:
    return " @EOF@ ".join([" @SEP@ ".join(t).replace('"', '') for t in triples])


def main(args: argparse.Namespace):
    with open(args.json_file) as f:
        doc = json.load(f)

    with open(args.out_file, 'w') as fout:
        for entry_id, entry in tqdm(enumerate(doc['entries'])):
            for e in entry.values():
                triples = []
                for triple in e['modifiedtripleset']:
                    triples.append((
                        preprocess_entity(triple['subject']),
                        preprocess_relation(triple['property']),
                        preprocess_entity(triple['object'])
                    ))
                graph = format_graph(triples)
                for lex in e['lexicalisations']:
                    text = lex['lex'].replace('"', '')
                    print(entry_id, text, 'graph', graph,
                          file=fout, sep='\t')
                    print(entry_id, graph, 'text', text,
                          file=fout, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('out_file')
    args = parser.parse_args()
    main(args)
