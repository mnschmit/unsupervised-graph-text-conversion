from typing import List

from allennlp.training.metrics.metric import Metric
from .ie_eval import factseq2set


@Metric.register("IE-f1")
class IE_F1Score(Metric):
    """
    Token-based Metric.
    Converts serialized graphs back to KG facts and
    computes Precision, Recall and F1 for the information extraction task.
    """

    def __init__(self) -> None:
        self._num_correct_retrieved = 0.0
        self._num_retrieved_facts = 0.0
        self._num_correct_facts = 0.0

    def __call__(self,
                 predictions: List[List[str]],
                 gold_seq: List[List[str]]
                 ):
        """
        Parameters
        ----------
        predictions : ``List[List[str]]``, required.
            A predicted sequence for each item in the batch.
        gold_seq : ``List[List[str]]``, required.
            A gold sequence for each item in the batch.
        """
        fact_pred = [
            factseq2set(predicted_seq)
            for predicted_seq in predictions
        ]
        fact_truth = [
            factseq2set(truth_seq)
            for truth_seq in gold_seq
        ]

        for pred_set, true_set in zip(fact_pred, fact_truth):
            self._num_retrieved_facts += len(pred_set)
            self._num_correct_facts += len(true_set)
            self._num_correct_retrieved = len(true_set & pred_set)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1 score : float
        """
        precision = float(self._num_correct_retrieved) / \
            float(self._num_retrieved_facts + 1e-13)
        recall = float(self._num_correct_retrieved) / \
            float(self._num_correct_facts + 1e-13)
        f1_score = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()

        return {"prec": precision, "rec": recall, "f1": f1_score}

    def reset(self):
        self._num_correct_retrieved = 0.0
        self._num_retrieved_facts = 0.0
        self._num_correct_facts = 0.0
