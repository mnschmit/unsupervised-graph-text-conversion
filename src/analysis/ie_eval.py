from typing import List, Tuple, FrozenSet


def generate_triple(fact_buffer: List[str], sep: str) -> Tuple[str, str, str]:
    triple_buffers = ([], [], [])
    curr_buffer = 0
    for elem in fact_buffer:
        if elem == sep:
            curr_buffer += 1
            if curr_buffer > 2:
                break
        else:
            triple_buffers[curr_buffer].append(elem)

    return tuple(" ".join(buf) for buf in triple_buffers)


def factseq2set(seq: List[str], eof="@EOF@", sep="@SEP@",
                lower=True) -> FrozenSet[Tuple[str, str, str]]:
    facts: List[Tuple[str, str, str]] = []
    fact_buffer: List[str] = []
    for token in seq:
        if token == eof:
            facts.append(generate_triple(fact_buffer, sep))
            fact_buffer.clear()
        else:
            fact_buffer.append(
                token.lower() if lower and token != sep else token)

    if fact_buffer:
        facts.append(generate_triple(fact_buffer, sep))

    return frozenset(facts)
