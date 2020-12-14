import argparse
import random


def main(args):
    with open(args.dev_file) as f:
        lines = f.readlines()
    rng = random.Random(0)
    sampled_lines = rng.sample(lines, 100)
    with open(args.out_file, 'w') as f:
        for line in sampled_lines:
            print(line, file=f, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dev_file')
    parser.add_argument('out_file')
    args = parser.parse_args()
    main(args)
