import argparse
import os
import logging
from .analysis.predict import main as generate_predictions


def main(args):
    logger = logging.getLogger(__name__)
    for iter_num in range(args.first_iter, args.last_iter+1):
        model_th = os.path.join(
            args.model_dir, 'iter{}'.format(iter_num), 'best.th')
        dataset = os.path.join(
            args.data_dir, 'iter{}-data.txt'.format(iter_num))
        hfn = os.path.join(args.data_dir, 'iter{}.txt'.format(iter_num))
        rfn = os.path.join(args.data_dir, 'iter{}-ref.txt'.format(iter_num))

        logger.warn('Start predictions on data from {}'.format(dataset))
        logger.warn('using model weights {}'.format(model_th))
        logger.warn('writing generations to {}'.format(hfn))
        logger.warn('writing references to {}'.format(rfn))
        generate_predictions(args.config, model_th, dataset,
                             hfn, rfn, args.batch_size, args.no_gpu)
        logger.warn('Finished predictions for iteration {}'.format(iter_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('model_dir')
    parser.add_argument('data_dir')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--first-iter', type=int, default=0)
    parser.add_argument('--last-iter', type=int, default=29)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    main(args)
