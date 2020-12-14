import csv
from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.copynet_seq2seq import CopyNetDatasetReader

from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("bt-supervised")
class SupervisedBTReader(CopyNetDatasetReader):
    """
    Same as `allennlp.data.dataset_readers.copynet_seq2seq.CopyNetDatasetReader`;
    only adds the boolean parameter `back_translation`.
    If True, source and target side are switched.

    Parameters
    ----------
    back_translation : bool, if True, switches source and target side
    """

    @overrides
    def __init__(
            self,
            target_namespace: str,
            source_tokenizer: Tokenizer = None,
            target_tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = None,
            delimiter: str = "\t",
            back_translation: bool = False,
            lazy: bool = False
    ) -> None:
        super().__init__(
            target_namespace,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_token_indexers=source_token_indexers,
            lazy=lazy
        )
        self._delimiter = delimiter
        self._back_translation = back_translation

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if len(row) != 3:
                    raise ConfigurationError(
                        "Invalid line format: {} (line number {})".format(
                            row, line_num + 1
                        )
                    )

                if self._back_translation:
                    image_id, target_sequence, source_sequence = row
                else:
                    image_id, source_sequence, target_sequence = row

                if not source_sequence:
                    raise ConfigurationError(
                        "Empty source sequence: {} (line number {})".format(
                            row, line_num + 1
                        )
                    )

                if not target_sequence:
                    yield self.text_to_instance(source_sequence)
                else:
                    yield self.text_to_instance(source_sequence, target_sequence)
