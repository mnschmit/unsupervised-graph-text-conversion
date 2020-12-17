import logging
from typing import Dict, List, Any

from overrides import overrides
import torch
from torch.nn import Dropout

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.models.encoder_decoders.copynet_seq2seq import CopyNetSeq2Seq
from allennlp.training.metrics.metric import Metric

# from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)


@Model.register("copynet_shared_decoder")
class CopyNetSharedDecoder(CopyNetSeq2Seq):
    """
    This is an implementation of `CopyNet <https://arxiv.org/pdf/1603.06393>`
    that can be used with multiple output languages.
    CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
    that can copy tokens from the source sentence into the target sentence instead of
    generating all target tokens only from the target vocabulary.
    The decoder is fed the first token of the target side without prediction
    to indicate the desired target language.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    attention : ``Attention``, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    beam_size : ``int``, required
        Beam width to use for beam search prediction.
    max_decoding_steps : ``int``, required
        Maximum sequence length of target predictions.
    dropout : ``float``, optional (default = 0.0)
        Encoder/decoder input dropout rate; default is no dropout.
    target_embedding_dim : ``int``, optional (default = 30)
        The size of the embeddings for the target vocabulary.
        If source and target namespace are the same, this is ignored and
        source and target words will share embeddings.
    copy_token : ``str``, optional (default = '@COPY@')
        The token used to indicate that a target token was copied from the source.
        If this token is not already in your target vocabulary, it will be added.
    source_namespace : ``str``, optional (default = 'source_tokens')
        The namespace for the source vocabulary.
    target_namespace : ``str``, optional (default = 'target_tokens')
        The namespace for the target vocabulary.
    language_id__namespace : ``str``, optional (default = 'language_labels')
        The namespace for the target language vocabulary.
    tensor_based_metric : ``Metric``, optional (default = BLEU)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : ``Metric``, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    initializer : ``InitializerApplicator``, optional
        An initialization strategy for the model weights.
    """

    @overrides
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 dropout: float = 0.0,
                 target_embedding_dim: int = 30,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 language_id_namespace: str = "language_labels",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        if source_namespace == target_namespace:
            target_embedding_dim = source_embedder._token_embedders[
                source_namespace
            ].get_output_dim()

        super().__init__(
            vocab, source_embedder, encoder, attention, beam_size,
            max_decoding_steps, target_embedding_dim=target_embedding_dim,
            copy_token=copy_token, source_namespace=source_namespace,
            target_namespace=target_namespace,
            tensor_based_metric=tensor_based_metric,
            token_based_metric=token_based_metric
        )
        self._language_id_namespace = language_id_namespace

        self.lang_vocab_size = self.vocab.get_vocab_size(
            self._language_id_namespace)
        self._lang_embedder = Embedding(
            self.lang_vocab_size, self.decoder_output_dim)

        self._inp_dropout = Dropout(p=dropout)

        if source_namespace == target_namespace:
            # replace independent target embeddings by source embeddings
            self._target_embedder = self._source_embedder._token_embedders[source_namespace]

        # self._bt_loss = Average()
        # self._lm_loss = Average()

        initializer(self)

    @staticmethod
    def _tokens_to_ids(batch: torch.LongTensor) -> List[List[int]]:
        batch_out = []
        for seq in batch:
            ids: Dict[torch.LongTensor, int] = {}
            out: List[int] = []
            for entry in seq:
                out.append(ids.setdefault(entry, len(ids)))
            batch_out.append(out)
        return batch_out

    @overrides
    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                source_token_ids: torch.Tensor,
                source_to_target: torch.Tensor,
                metadata: List[Dict[str, Any]],
                target_lang: torch.Tensor,
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_token_ids: torch.Tensor = None,
                target_language: List[str] = None,
                dataset: List[str] = None,
                epoch_num: List[int] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Make foward pass with decoder logic for producing the entire target sequence.
        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``, required
            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        source_token_ids : ``torch.Tensor``, required
            Tensor containing IDs that indicate which source tokens match each other.
            Has shape: `(batch_size, trimmed_source_length)`.
        source_to_target : ``torch.Tensor``, required
            Tensor containing vocab index of each source token with respect to the
            target vocab namespace. Shape: `(batch_size, trimmed_source_length)`.
        metadata : ``List[Dict[str, Any]]``, required
            Metadata field that contains the original source tokens with key 'source_tokens'
            and any other meta fields. When 'target_tokens' is also passed, the metadata
            should also contain the original target tokens with key 'target_tokens'.
        target_lang : `torch.Tensor`, required
            The target language id labels for the current batch.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField` which must contain a "tokens"
            key that uses single ids.
        target_token_ids : ``torch.Tensor``, optional (default = None)
            A tensor of shape `(batch_size, target_sequence_length)` which indicates which
            tokens in the target sequence match tokens in the source sequence.
        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode(source_tokens)
        state["source_token_ids"] = source_token_ids
        state["source_to_target"] = source_to_target
        state["target_lang"] = self._lang_embedder(target_lang)

        if target_tokens:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(
                target_tokens, target_token_ids, state)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

        # if all([d == "reconstruction" for d in dataset]):
        #     self._lm_loss(output_dict["loss"])

        # if self.training and (
        #         epoch_num is not None and epoch_num[0] > 1
        # ) and (
        #     dataset is None or
        #     all([d == "supervised" for d in dataset])
        # ):
        #     bt_output_dict = self._backtranslation_loss(
        #         state, source_tokens, source_token_ids, target_token_ids,
        #         target_lang
        #     )
        #     output_dict["loss"] += bt_output_dict["loss"]
        #     self._bt_loss(bt_output_dict["loss"])

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]
                    # shape: (batch_size, target_sequence_length)
                    gold_tokens = self._gather_extended_gold_tokens(
                        target_tokens["tokens"],
                        source_token_ids,
                        target_token_ids
                    )
                    self._tensor_based_metric(best_predictions, gold_tokens)

                if self._token_based_metric is not None:
                    predicted_tokens = self._get_predicted_tokens(
                        output_dict["predictions"],
                        metadata,
                        n_best=1
                    )
                    self._token_based_metric(
                        predicted_tokens,
                        [x["target_tokens"] for x in metadata]
                    )

        return output_dict

    def _backtranslation_loss(self, state, source_tokens, source_token_ids,
                              target_token_ids, target_lang):
        # obtain backtranslation prediction
        self.eval()
        with torch.no_grad():
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            best_predictions = predictions["predictions"][:, 0, :]
        self.train()

        # repair best_predictions
        for t in best_predictions:
            write_zeros = False
            for i, e in enumerate(t):
                if write_zeros:
                    t[i] = 0
                else:
                    if e == 0:
                        write_zeros = True

        # construct new "source tokens"
        batch_size, max_pred_seq_length = best_predictions.size()
        starters = best_predictions.new_full(
            (batch_size, 1), fill_value=self._start_index
        )
        end_zeros = best_predictions.new_zeros(
            (batch_size, 1)
        )
        new_source_tokens = torch.cat(
            [starters, best_predictions, end_zeros], 1
        )
        for t in new_source_tokens:
            for i, e in enumerate(t):
                if e == 0:
                    t[i] = self._end_index
                    break

        # construct new source and target token ids
        new_source_and_target_token_ids = self._tokens_to_ids(
            torch.cat([best_predictions, source_tokens["tokens"]], 1)
        )
        new_source_token_ids = []
        new_target_token_ids = []
        for seq in new_source_and_target_token_ids:
            new_source_token_ids.append(seq[:max_pred_seq_length])
            new_target_token_ids.append(seq[max_pred_seq_length:])
        new_source_token_ids = torch.as_tensor(
            new_source_token_ids, device=source_token_ids.device
        )
        new_target_token_ids = torch.as_tensor(
            new_target_token_ids, device=target_token_ids.device
        )

        # train with the backtranslated samples
        bt_state = self._encode({"tokens": new_source_tokens})
        bt_state["source_token_ids"] = new_source_token_ids
        bt_state["source_to_target"] = best_predictions
        bt_state["target_lang"] = self._lang_embedder(
            1 - target_lang
        )
        bt_state = self._init_decoder_state(bt_state)
        bt_output_dict = self._forward_loss(
            source_tokens, new_target_token_ids, bt_state
        )

        bt_output_dict["intermediate"] = best_predictions

        return bt_output_dict

    @overrides
    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        batch_size, _ = state["source_mask"].size()

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"],
            state["source_mask"],
            self._encoder.is_bidirectional())
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["target_lang"]
        # ORIGINAL: state["encoder_outputs"].new_zeros(batch_size, self.decoder_output_dim)

        return state

    @overrides
    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)

        # for word dropout (remember to check for model.train/eval)
        # torch.where(probs > 0.2, x, torch.zeros(10, dtype=torch.int64))

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input_after_dropout = self._inp_dropout(embedded_input)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(
            embedded_input_after_dropout, source_mask
        )

        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    @overrides
    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      selective_weights: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"].float()
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._inp_dropout(
            self._target_embedder(last_predictions)
        )
        # shape: (group_size, max_input_sequence_length)
        attentive_weights = self._attention(
            state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask
        )
        # shape: (group_size, encoder_output_dim)
        attentive_read = util.weighted_sum(
            state["encoder_outputs"], attentive_weights
        )
        # shape: (group_size, encoder_output_dim)
        selective_read = util.weighted_sum(
            state["encoder_outputs"][:, 1:-1], selective_weights
        )
        # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat(
            (embedded_input, attentive_read, selective_read), -1
        )
        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
            projected_decoder_input,
            (state["decoder_hidden"], state["decoder_context"])
        )
        return state

    # @overrides
    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     all_metrics: Dict[str, float] = super().get_metrics(reset=reset)
    #     all_metrics["bt_loss"] = self._bt_loss.get_metric(reset=reset)
    #     all_metrics["lm_loss"] = self._lm_loss.get_metric(reset=reset)
    #     return all_metrics
