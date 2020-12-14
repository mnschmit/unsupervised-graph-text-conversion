from commandr import Run, command
import spacy
from tqdm import tqdm


def get_graph_tokens(graph_str, tokenizer):
    tokens = []
    for fact_str in graph_str.split(' @EOF@ '):
        for elem in fact_str.split(' @SEP@ '):
            tokens.extend(get_text_tokens(elem, tokenizer))
    return tokens


def get_text_tokens(text_str, tokenizer):
    return [tok.text for tok in tokenizer(text_str)]


@command
def text_in_graph(datasets=[]):
    tokenizer = get_tokenizer()

    percentage_sum = 0
    num_samples = 0
    for d in datasets:
        with open(d) as f:
            for line in tqdm(f):
                img_id, graph, lang, text = line.strip().split('\t')
                assert lang == 'text'
                graph_tokens = set(get_graph_tokens(graph, tokenizer))
                text_tokens = get_text_tokens(text, tokenizer)
                num_tokens = len(text_tokens)
                num_tok_in_graph = 0
                for t in text_tokens:
                    if t in graph_tokens:
                        num_tok_in_graph += 1
                percentage_sum += num_tok_in_graph / num_tokens * 100
                num_samples += 1
    print("Avg % text tokens in graph input: {:.1f}".format(
        percentage_sum/num_samples))


@command
def graph_in_text(datasets=[]):
    tokenizer = get_tokenizer()

    percentage_sum = 0
    num_samples = 0
    for d in datasets:
        with open(d) as f:
            for line in tqdm(f):
                img_id, text, lang, graph = line.strip().split('\t')
                assert lang == 'graph'
                graph_tokens = get_graph_tokens(graph, tokenizer)
                text_tokens = set(get_text_tokens(text, tokenizer))
                num_tokens = len(graph_tokens)
                num_tok_in_text = 0
                for t in graph_tokens:
                    if t in text_tokens:
                        num_tok_in_text += 1
                percentage_sum += num_tok_in_text / num_tokens * 100
                num_samples += 1
    print("Avg % graph tokens in text input: {:.1f}".format(
        percentage_sum/num_samples))


def get_tokenizer():
    return spacy.load('en_core_web_sm', disable=[
        'vectors', 'textcat', 'tagger', 'parser', 'ner'])


@command
def textlength(datasets=[], col=1):
    tokenizer = get_tokenizer()

    num_tokens = 0
    num_texts = 0
    for dataset in datasets:
        with open(dataset) as f:
            for line in tqdm(f):
                text = line.strip().split('\t')[col]
                analyzed = tokenizer(text)
                for token in analyzed:
                    num_tokens += 1
                num_texts += 1
    print('Avg text length: {:.1f}'.format(num_tokens / num_texts))


@command
def facts(datasets=[], col=1):
    num_facts = 0
    num_graphs = 0
    for dataset in datasets:
        with open(dataset) as f:
            for line in tqdm(f):
                graph = line.strip().split('\t')[col]
                num_seps = graph.count('@SEP@')
                assert num_seps % 2 == 0
                num_facts += num_seps // 2
                num_graphs += 1
    print('Avg #facts: {:.1f}'.format(num_facts / num_graphs))


if __name__ == '__main__':
    Run()
