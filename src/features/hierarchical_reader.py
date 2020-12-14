import logging
from typing import List, Dict

import numpy as np
from overrides import overrides

import csv

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.copynet_seq2seq import CopyNetDatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from .make_noise import noise_fun
import random

logger = logging.getLogger(__name__)


@DatasetReader.register("hierarchical_seq_reader")
class HierarchicalSeqDatasetReader(CopyNetDatasetReader):
    """
    Reads a tsv file of the following format:
    image_id TAB source_seq TAB target_lang TAB target_seq
    where target_seq maybe empty but none of the others
    and produces a dataset suitable for the ``CopyNetSelfAttention`` model
    or any model with a matching API.

    An instance produced by ``HierarchicalSeqDatasetReader`` will contain at least the following fields:
    - ``source_tokens``: a ``ListField[TextField]`` containing a list of 
       tokenized source sentences
       (single entry for natural language, multiple ones only for facts),
       including the ``START_SYMBOL`` and ``END_SYMBOL`` (for each entry).
       This will result in a tensor of shape ``(batch_size, num_entries, source_length)``.
    - ``source_token_ids``: an ``ArrayField`` of size ``(batch_size, trimmed_source_length)``
      that contains an ID for each token in the source sentence. Tokens that
      match at the lowercase level will share the same ID. If ``target_tokens``
      is passed as well, these IDs will also correspond to the ``target_token_ids``
      field, i.e. any tokens that match at the lowercase level in both
      the source and target sentences will share the same ID. Note that these IDs
      have no correlation with the token indices from the corresponding
      vocabulary namespaces.
    - ``source_to_target``: a ``NamespaceSwappingField`` that keeps track of the index
      of the target token that matches each token in the source sentence.
      When there is no matching target token, the OOV index is used.
      This will result in a tensor of shape ``(batch_size, trimmed_source_length)``.
    - ``metadata``: a ``MetadataField`` which contains the source tokens and
      potentially target tokens as lists of strings.
    - ``target_lang``: a ``LabelField`` that contains the language identifier for the desired output language.
    - ``target_language``: a ``MetadataField`` that contains the language identifier for the desired output language. It can be used to sort samples by language to obtain homogenous batches.
    When ``target_string`` is passed, the instance will also contain these fields:
    - ``target_tokens``: a ``TextField`` containing the tokenized target sentence,
      including the ``START_SYMBOL`` and ``END_SYMBOL``. This will result in
      a tensor of shape ``(batch_size, target_length)``.
    - ``target_token_ids``: an ``ArrayField`` of size ``(batch_size, target_length)``.
      This is calculated in the same way as ``source_token_ids``.

    See the "Notes" section below for a description of how these fields are used.
    Parameters
    ----------
    target_namespace : ``str``, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the NamespaceSwappingField.
    language_id_namespace : ``str``, optional
        The vocab namespace for the target language identifiers. Defaults to ``language_labels``.
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to ``source_token_indexers``.

    Notes
    -----
    By ``source_length`` we are referring to the number of tokens in the source
    sentence including the ``START_SYMBOL`` and ``END_SYMBOL``, while
    ``trimmed_source_length`` refers to the number of tokens in the source sentence
    *excluding* the ``START_SYMBOL`` and ``END_SYMBOL``, i.e.
    ``trimmed_source_length = source_length - 2``.
    On the other hand, ``target_length`` is the number of tokens in the target sentence
    *including* the ``START_SYMBOL`` and ``END_SYMBOL``.
    In the context where there is a ``batch_size`` dimension, the above refer
    to the maximum of their individual values across the batch.
    In regards to the fields in an ``Instance`` produced by this dataset reader,
    ``source_token_ids`` and ``target_token_ids`` are primarily used during training
    to determine whether a target token is copied from a source token (or multiple matching
    source tokens), while ``source_to_target`` is primarily used during prediction
    to combine the copy scores of source tokens with the generation scores for matching
    tokens in the target namespace.
    """

    def __init__(self,
                 target_namespace: str,
                 language_id_namespace: str = "language_labels",
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 provide_target_lang: bool = True,
                 lazy: bool = False
                 ) -> None:
        super().__init__(
            target_namespace, source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_token_indexers=source_token_indexers, lazy=lazy
        )

        self._provide_trg_lang = provide_target_lang
        self._language_id_namespace = language_id_namespace

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)
            reader = csv.reader(data_file, delimiter="\t")

            line_num: int
            row: List[str]
            for line_num, row in enumerate(reader):
                if len(row) != 4:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1))
                sample_id, source_sequence, target_lang, target_sequence = row

                if not target_lang:
                    raise ConfigurationError(
                        "Empty target language: {} (line number {})".format(row, line_num + 1))

                yield self.text_to_instance(
                    source_sequence, target_lang, target_sequence
                )

    @overrides
    def text_to_instance(self, source_string: str, target_lang: str,
                         target_string: str = None) -> Instance:
        """
        Turn raw source string and target string into an ``Instance``.
        Parameters
        ----------
        source_string : ``str``, required
        target_lang : ``str``, required
        target_string : ``str``, optional (default = None)
        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(
            tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [
            x.text for x in tokenized_source[1:-1]]}

        fields_dict = {
            "source_tokens": source_field,
            "source_to_target": source_to_target_field,
        }

        if self._provide_trg_lang:
            lang_id_field = LabelField(
                target_lang, label_namespace=self._language_id_namespace
            )
            metadata_trg_lang = MetadataField(target_lang)

            fields_dict["target_lang"] = lang_id_field
            fields_dict["target_language"] = metadata_trg_lang

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(
                tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [
                y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(
                tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(
                np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(
                tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(
                np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(
                np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)


class NoisyDatasetReader(CopyNetSharedDecoderDatasetReader):
    """
    Abstract class that adds `noises` as a parameter to the constructor.
    """

    def __init__(
            self,
            target_namespace: str,
            language_id_namespace: str = "language_labels",
            source_tokenizer: Tokenizer = None,
            target_tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = None,
            noises: List[str] = ["swap", "drop", "blank", "rule", "repeat"],
            provide_target_lang: bool = True,
            use_all_noise: bool = False,
            all_for_one: bool = False,
            no_noise: bool = False,
            lazy: bool = True) -> None:
        super().__init__(
            target_namespace,
            language_id_namespace=language_id_namespace,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_token_indexers=source_token_indexers,
            provide_target_lang=provide_target_lang,
            lazy=lazy
        )

        self._noises = noises
        self.flip_trg_lang = {
            "graph": "text",
            "text": "graph"
        }
        self._use_all_noise = use_all_noise
        self._all_for_one = all_for_one
        self._no_noise = no_noise
        assert not (use_all_noise and all_for_one)
        assert not (use_all_noise and no_noise)
        assert not (all_for_one and no_noise)


@DatasetReader.register("shared_bt_reconstruction")
class ReconstructionDatasetReader(NoisyDatasetReader):
    """
    Same as `CopyNetSharedDecoderDatasetReader`.
    Only yields instances according to reconstruction loss.
    """

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)
            reader = csv.reader(data_file, delimiter="\t")

            line_num: int
            row: List[str]
            for line_num, row in enumerate(reader):
                if len(row) != 4:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1))
                sample_id, source_sequence, target_lang, target_sequence = row

                if not target_lang:
                    raise ConfigurationError(
                        "Empty target language: {} (line number {})".format(row, line_num + 1))

                if self._use_all_noise:
                    for noise in self._noises:
                        noised_src, _ = noise_fun[noise](
                            source_sequence, target_lang == "text"
                        )
                        yield self.text_to_instance(
                            noised_src, self.flip_trg_lang[target_lang],
                            source_sequence
                        )
                elif self._no_noise:
                    yield self.text_to_instance(
                        source_sequence, self.flip_trg_lang[target_lang],
                        source_sequence
                    )
                else:
                    is_graph = target_lang == "text"
                    if self._all_for_one:
                        noised_src = source_sequence
                        for noise in self._noises:
                            noised_src, is_graph = noise_fun[noise](
                                noised_src, is_graph
                            )
                    else:
                        noise = random.choice(self._noises)
                        noised_src, _ = noise_fun[noise](
                            source_sequence, is_graph
                        )

                    yield self.text_to_instance(
                        noised_src, self.flip_trg_lang[target_lang],
                        source_sequence
                    )


@DatasetReader.register("shared_bt_noise_sup")
class NoisySupervisedDatasetReader(NoisyDatasetReader):
    """
    Same as `CopyNetSharedDecoderDatasetReader`.
    Only yields instances according to noisy supervised loss.
    """

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)
            reader = csv.reader(data_file, delimiter="\t")

            line_num: int
            row: List[str]
            for line_num, row in enumerate(reader):
                if len(row) != 4:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1))
                sample_id, source_sequence, target_lang, target_sequence = row

                if not target_lang:
                    raise ConfigurationError(
                        "Empty target language: {} (line number {})".format(row, line_num + 1))

                is_graph = target_lang == "text"
                if self._all_for_one:
                    noised_src = source_sequence
                    for noise in self._noises:
                        noised_src, is_graph = noise_fun[noise](
                            noised_src, is_graph
                        )
                else:
                    noise = random.choice(self._noises)
                    noised_src, _ = noise_fun[noise](
                        source_sequence, target_lang == "text"
                    )

                yield self.text_to_instance(
                    noised_src, target_lang, target_sequence
                )
