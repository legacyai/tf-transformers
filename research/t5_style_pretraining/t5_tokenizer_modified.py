# -*- coding: utf-8 -*-
# mypy: ignore-errors

# coding=utf-8
# Copyright 2021 TF-Transformers Authors and The TensorFlow Authors.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""T5 Tokenizer based on TFText"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Union

import sentencepiece
import tensorflow as tf
import tensorflow_text as tf_text
from absl import logging

_PREFIX_DIR = 'tftransformers_tokenizer_cache'

code_example = r'''

        >>> from tf_transformers.models import  T5TokenizerTFText
        >>> tokenizer = T5TokenizerTFText.from_pretrained("t5-small")
        >>> text = ['The following statements are true about sentences in English:',
                    '',
                    'A new sentence begins with a capital letter.']
        >>> inputs = {'text': text}
        >>> outputs = tokenizer(inputs) # Ragged Tensor Output

        # Dynamic Padding
        >>> tokenizer = T5TokenizerTFText.from_pretrained("t5-small", dynamic_padding=True)
        >>> text = ['The following statements are true about sentences in English:',
                    '',
                    'A new sentence begins with a capital letter.']
        >>> inputs = {'text': text}
        >>> outputs = tokenizer(inputs) # Dict of tf.Tensor

        # Static Padding
        >>> tokenizer = T5TokenizerTFText.from_pretrained("t5-small", pack_model_inputs=True)
        >>> text = ['The following statements are true about sentences in English:',
                    '',
                    'A new sentence begins with a capital letter.']
        >>> inputs = {'text': text}
        >>> outputs = tokenizer(inputs) # Dict of tf.Tensor

        # To Add Special Tokens
        >>> tokenizer = T5TokenizerTFText.from_pretrained("t5-small", add_special_tokens=True)

'''


def get_vocab(model_proto):
    """Get vocab from sentencpiece model"""
    sp_model = sentencepiece.SentencePieceProcessor()
    sp_model.LoadFromSerializedProto(model_proto)
    vocab = {sp_model.IdToPiece(i): i for i in range(sp_model.GetPieceSize())}
    return vocab


class T5TokenizerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        lower_case,
        model_file_path=None,
        model_serialized_proto=None,
        out_type=tf.int32,
        tokenize_with_offsets=False,
        nbest_size: int = 0,
        alpha: float = 1.0,
        strip_diacritics=False,
        cls_enc_token_id=None,
        cls_dec_token_id=None,
        sep_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        unk_token_id=None,
        pad_token_id=None,
        mask_token_id=None,
        max_length=512,
        add_special_tokens=False,
        pack_model_inputs=False,
        dynamic_padding=False,
        truncate=False,
        **kwargs,
    ):
        """Initializes a SentencepieceTokenizer layer.
        Args:
            lower_case: A Python boolean indicating whether to lowercase the string
                    before tokenization. NOTE: New models are encouraged to build `*_cf`
                    (case folding) normalization into the Sentencepiece model itself and
                    avoid this extra step.
            special_tokens: A list of special tokens , must present in model. If not pass None.
                    model_file_path: A Python string with the path of the sentencepiece model.
                    Exactly one of `model_file_path` and `model_serialized_proto` can be
                    specified. In either case, the Keras model config for this layer will
                    store the actual proto (not a filename passed here).
                    model_serialized_proto: The sentencepiece model serialized proto string.
            add_cls_sep: To add [CLS] and [SEP] with the tokenized text
            cls_token: cls token string
            sep_token: sep token string
            tokenize_with_offsets: A Python boolean. If true, this layer calls
                    `SentencepieceTokenizer.tokenize_with_offsets()` instead of
                    plain `.tokenize()` and outputs a triple of
                    `(tokens, start_offsets, limit_offsets)` insead of just tokens.
                    Note that when following `strip_diacritics` is set to True, returning
                    offsets is not supported now.
            nbest_size: A scalar for sampling:
            nbest_size = {0,1}: No sampling is performed. (default)
            nbest_size > 1: samples from the nbest_size results.
            nbest_size < 0: assuming that nbest_size is infinite and samples
                    from the all hypothesis (lattice) using
                    forward-filtering-and-backward-sampling algorithm.
            alpha: A scalar for a smoothing parameter. Inverse temperature for
                    probability rescaling.
            strip_diacritics: Whether to strip diacritics or not. Note that stripping
                    diacritics requires additional text normalization and dropping bytes,
                    which makes it impossible to keep track of the offsets now. Hence
                    when `strip_diacritics` is set to True, we don't yet support
                    `tokenize_with_offsets`. NOTE: New models are encouraged to put this
                    into custom normalization rules for the Sentencepiece model itself to
                    avoid this extra step and the limitation regarding offsets.
            **kwargs: standard arguments to `Layer()`.
            add_special_tokens: If True: Add special tokens CLS and SEP.
            pack_model_inputs: Static Padding to max_length
            dynamic_padding: Dynamic Padding to max_length of the batch
            truncate: To enable truncate

        Raises:
            ImportError: if importing tensorflow_text failed.

        Returns:
            Default: RaggedTensor
            if dynamic_padding or bert_pack_inputs: dict of tf.Tensor


        """

        super().__init__(**kwargs)
        if bool(model_file_path) == bool(model_serialized_proto):
            raise ValueError("Exact one of `model_file_path` and " "`model_serialized_proto` can be specified.")
        # TODO(b/181866850): Support tokenize_with_offsets for strip_diacritics=True
        if tokenize_with_offsets and strip_diacritics:
            raise ValueError("`tokenize_with_offsets` is not supported when " "`strip_diacritics` is set to True.")
        if model_file_path:
            self._model_serialized_proto = tf.io.gfile.GFile(model_file_path, "rb").read()
        else:
            self._model_serialized_proto = model_serialized_proto

        self._vocab = get_vocab(self._model_serialized_proto)
        self._lower_case = lower_case

        # self.add_cls_sep = add_cls_sep
        self.tokenize_with_offsets = tokenize_with_offsets

        self._nbest_size = nbest_size
        self._alpha = alpha
        self._strip_diacritics = strip_diacritics
        self.out_type = out_type

        self._tokenizer = self._create_tokenizer()

        # Tokenizer specifics
        self.cls_enc_token_id = cls_enc_token_id
        self.cls_dec_token_id = cls_dec_token_id
        self.sep_token_id = sep_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        self.max_length = max_length

        self.add_special_tokens = add_special_tokens
        self.pack_model_inputs = pack_model_inputs
        self.dynamic_padding = dynamic_padding
        self.truncate = truncate

        if self.dynamic_padding and self.pack_model_inputs:
            raise ValueError("Either dynamic_padding is True, or pack_model_inputs is True. Don't set them together")

    def _create_tokenizer(self):
        """Return sentencepiece tokenizer."""
        return tf_text.SentencepieceTokenizer(
            model=self._model_serialized_proto, out_type=self.out_type, nbest_size=self._nbest_size, alpha=self._alpha
        )

    @property
    def vocab_size(self):
        """Return vocab size."""
        return self._tokenizer.vocab_size()

    def encode(
        self,
        inputs: Dict[str, tf.Tensor],
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int = None,
        padding: bool = False,
        max_pad_length: bool = None,
        return_tensors: str = 'ragged',
    ):
        """Calls `text.SentencepieceTokenizer` on inputs.
        Args:
            inputs: A string Tensor of shape `(batch_size,)`.
        Returns:
            One or three of RaggedTensors if tokenize_with_offsets is False or True,
            respectively. These are
            tokens: A RaggedTensor of shape `[batch_size, (pieces)]` and type `int32`.
            `tokens[i,j]` contains the j-th piece in the i-th input.
            start_offsets, limit_offsets: If `tokenize_with_offsets` is True,
            RaggedTensors of type `int64` with the same indices as tokens.
            Element `[i,j]` contains the byte offset at the start, or past the
            end, resp., for the j-th piece in the i-th input.
        """

        inputs = tf.squeeze(inputs['text'], axis=0)
        if self._strip_diacritics:
            if self.tokenize_with_offsets:
                raise ValueError(
                    "`tokenize_with_offsets` is not supported yet when "
                    "`strip_diacritics` is set to True (b/181866850)."
                )
            inputs = tf_text.normalize_utf8(inputs, "NFD")
            inputs = tf.strings.regex_replace(inputs, r"\p{Mn}", "")

        if self._lower_case:
            inputs = tf_text.case_fold_utf8(inputs)

        # Prepare to reshape the result to work around broken shape inference.
        def _reshape(rt):
            values = rt.values
            row_splits = rt.row_splits
            row_splits = tf.reshape(row_splits, [batch_size + 1])
            return tf.RaggedTensor.from_row_splits(values, row_splits)

        batch_size = tf.shape(inputs)[0]
        tokens = self._tokenizer.tokenize(inputs)
        tokens = _reshape(tokens)
        # If Truncation is True
        if truncation:
            if max_length:
                tokens = tokens[:, : self.max_length]
            else:
                if add_special_tokens:
                    tokens = tokens[:, : self.max_length - 2]  # For cls and sep
                else:
                    tokens = tokens[:, : self.max_length]
        # If add special_tokens
        if add_special_tokens:
            tokens = self._add_special_tokens(tokens)

        input_mask = tf.ones_like(tokens)
        input_type_ids = tf.zeros_like(tokens)

        # If padding
        if padding:
            if max_pad_length is None:
                raise ValueError("If padding is True, please set max_pad_length")
            if isinstance(tokens, tf.RaggedTensor):
                tokens = tokens.to_tensor(default_value=self.pad_token_id, shape=(batch_size, max_pad_length))
                input_mask = input_mask.to_tensor(default_value=0, shape=(batch_size, max_pad_length))
                input_type_ids = input_type_ids.to_tensor(default_value=0, shape=(batch_size, max_pad_length))
        # If return_tensors='tf'
        if return_tensors == 'tf':
            if isinstance(tokens, tf.RaggedTensor):
                tokens = tokens.to_tensor(default_value=self.pad_token_id)
                input_mask = input_mask.to_tensor(default_value=0)
                input_type_ids = input_type_ids.to_tensor(default_value=0)

        result = {}
        result['input_ids'] = tokens
        result['input_mask'] = input_mask
        result['input_type_ids'] = input_type_ids

        return result

    def bert_pack_inputs(
        self,
        inputs: Union[tf.RaggedTensor, List[tf.RaggedTensor]],
        seq_length: Union[int, tf.Tensor],
        start_of_sequence_id: Union[int, tf.Tensor],
        end_of_segment_id: Union[int, tf.Tensor],
        padding_id: Union[int, tf.Tensor],
        num_special_tokens: int = 1,  # For EOS
        truncator="round_robin",
    ):
        """Freestanding equivalent of the BertPackInputs layer."""
        # _check_if_tf_text_installed()
        # Sanitize inputs.
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not inputs:
            raise ValueError("At least one input is required for packing")
        input_ranks = [rt.shape.rank for rt in inputs]
        if None in input_ranks or len(set(input_ranks)) > 1:
            raise ValueError(
                "All inputs for packing must have the same known rank, " "found ranks " + ",".join(input_ranks)
            )
        # Flatten inputs to [batch_size, (tokens)].
        if input_ranks[0] > 2:
            inputs = [rt.merge_dims(1, -1) for rt in inputs]
        # In case inputs weren't truncated (as they should have been),
        # fall back to some ad-hoc truncation.
        num_special_tokens = len(inputs) + 1
        if truncator == "round_robin":
            trimmed_segments = tf_text.RoundRobinTrimmer(seq_length - num_special_tokens).trim(inputs)
        elif truncator == "waterfall":
            trimmed_segments = tf_text.WaterfallTrimmer(seq_length - num_special_tokens).trim(inputs)
        else:
            raise ValueError("Unsupported truncator: %s" % truncator)
        # Combine segments.
        segments_combined, segment_ids = tf_text.combine_segments(
            trimmed_segments, start_of_sequence_id=start_of_sequence_id, end_of_segment_id=end_of_segment_id
        )
        if self.add_special_tokens is False:
            # Ignore cls token
            segments_combined = segments_combined[:, 1:-1]
            segment_ids = segment_ids[:, 1:-1]
        # Pad to dense Tensors.
        input_word_ids, _ = tf_text.pad_model_inputs(segments_combined, seq_length, pad_value=padding_id)
        input_type_ids, input_mask = tf_text.pad_model_inputs(segment_ids, seq_length, pad_value=0)
        # Work around broken shape inference.
        output_shape = tf.stack([inputs[0].nrows(out_type=tf.int32), tf.cast(seq_length, dtype=tf.int32)])  # batch_size

        def _reshape(t):
            return tf.reshape(t, output_shape)

        # Assemble nest of input tensors as expected by BERT TransformerEncoder.
        return dict(
            input_ids=_reshape(input_word_ids), input_mask=_reshape(input_mask), input_type_ids=_reshape(input_type_ids)
        )

    def preprocess(self, inputs):
        """Preprocess text like normalization, lower case etc"""
        if self._strip_diacritics:
            if self.tokenize_with_offsets:
                raise ValueError(
                    "`tokenize_with_offsets` is not supported yet when "
                    "`strip_diacritics` is set to True (b/181866850)."
                )
            inputs = tf_text.normalize_utf8(inputs, "NFD")
            inputs = tf.strings.regex_replace(inputs, r"\p{Mn}", "")

        if self._lower_case:
            inputs = tf_text.case_fold_utf8(inputs)
        return inputs

    def call_encode(self, inputs: Dict[str, tf.Tensor]):
        """Encode function used for serialization"""

        inputs = inputs['text']  # Get from dictionary
        inputs = self.preprocess(inputs)

        # Prepare to reshape the result to work around broken shape inference.
        batch_size = tf.shape(inputs)[0]

        def _reshape(rt):
            values = rt.values
            row_splits = rt.row_splits
            row_splits = tf.reshape(row_splits, [batch_size + 1])
            return tf.RaggedTensor.from_row_splits(values, row_splits)

        # Call the tokenizer.
        if self.tokenize_with_offsets:
            tokens, start_offsets, limit_offsets = self._tokenizer.tokenize_with_offsets(inputs)
            return _reshape(tokens), _reshape(start_offsets), _reshape(limit_offsets)
        else:
            tokens = self._tokenizer.tokenize(inputs)
            tokens = _reshape(tokens)
            # If add special_tokens
            if self.add_special_tokens:
                if self.truncate:
                    tokens = tokens[:, : self.max_length - 2]
                tokens = self._add_special_tokens(tokens)
            else:
                if self.truncate:
                    tokens = tokens[:, : self.max_length]

            if self.dynamic_padding:
                input_mask = tf.ones_like(tokens)
                input_word_ids = tokens.to_tensor(default_value=self.pad_token_id)
                input_mask = input_mask.to_tensor(default_value=0)
                # input_type_ids = tf.zeros_like(input_mask)
                seq_length = tf.shape(input_word_ids)[1]

                # Work around broken shape inference.
                output_shape = tf.stack(
                    [tokens.nrows(out_type=tf.int32), tf.cast(seq_length, dtype=tf.int32)]
                )  # batch_size

                def _reshape(t):
                    return tf.reshape(t, output_shape)

                # Assemble nest of input tensors as expected by BERT TransformerEncoder.
                return dict(encoder_input_ids=_reshape(input_word_ids), encoder_input_mask=_reshape(input_mask))

            if self.pack_model_inputs:
                tokens_dict = self.bert_pack_inputs(
                    tokens,
                    seq_length=self.max_length,
                    start_of_sequence_id=self.cls_token_id,
                    end_of_segment_id=self.sep_token_id,
                    padding_id=self.pad_token_id,
                )
                return tokens_dict
            else:
                return tokens

    def decode(self, inputs: Union[tf.Tensor, tf.RaggedTensor]):
        """Encode function used for serialization"""
        decoded_tokens = self._tokenizer.detokenize(inputs)
        return decoded_tokens

    def call(self, inputs):
        """Call"""
        results = self.call_encode(inputs)
        return results

    def get_config(self):
        # Skip in tf.saved_model.save(); fail if called direcly.
        # raise NotImplementedError("TODO(b/170480226): implement")
        return {}

    def _add_special_tokens(self, tokens):
        """Add special tokens"""
        #         batch_size = tokens.shape[0]
        #         print(batch_size, self.eos_token_id)
        #         eos_tokens_batch = tf.cast(tf.ones(shape=(batch_size, 1)), self.out_type) * self.eos_token_id
        #         tokens = tf.concat([tokens, eos_tokens_batch], axis=1)
        #         return tokens

        # -1 is dummy here not to break the tf_text ops
        tokens = tf_text.combine_segments([tokens], start_of_sequence_id=-1, end_of_segment_id=self.eos_token_id)[0]
        # return without -1 (0 th entry)
        return tokens[:, 1:]

    def get_model(self):
        """Convert Keras Layer to Model"""
        from tf_transformers.core import LegacyModel

        inputs = {
            "text": tf.keras.layers.Input(
                shape=(),
                dtype=tf.string,
                name="text",
            )
        }

        layer_outputs = self(inputs)
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name='albert_sentencepiece')
        return model


def create_cache_dir(cache_path: Path):
    """Create Cache Directory

    Args:
        cache_path : Path
    """
    if not cache_path.exists():  # If cache path not exists
        cache_path.mkdir()


def post_process_and_write(cache_path, special_tokens):
    from sentencepiece import sentencepiece_model_pb2

    m = sentencepiece_model_pb2.ModelProto()
    m.ParseFromString(open(os.path.join(cache_path, 'spiece.model'), "rb").read())

    for token in special_tokens:
        new_token = sentencepiece_model_pb2.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        m.pieces.append(new_token)

    with open(os.path.join(cache_path, 'spiece.model'), 'wb') as f:
        f.write(m.SerializeToString())


class T5CustomTokenizerTFText:
    def __init__(self) -> None:
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        out_type=tf.int32,
        max_length=None,
        add_special_tokens=False,
        pack_model_inputs=False,
        dynamic_padding=False,
        truncate=False,
    ):
        """Load HuggingFace tokenizer and pass to TFtext"""
        cache_dir = tempfile.gettempdir()
        cache_dir = Path(cache_dir, _PREFIX_DIR)
        create_cache_dir(cache_dir)

        cache_path = Path(cache_dir, model_name)

        from transformers import T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(model_name)
        if not cache_path.exists():
            tokenizer.save_pretrained(str(cache_path))
            logging.info("Saving {} tokenizer to {}".format(model_name, cache_path))

            # Adding [MASK] as special token
            special_tokens = ['[CLS_ENC]', '[MASK], [CLS_DEC]']
            post_process_and_write(cache_path, special_tokens)

        if max_length is None:
            max_length = tokenizer.max_len_single_sentence
        spiece_model = str(Path(cache_path, 'spiece.model'))
        logging.info("Loading {} tokenizer to {}".format(model_name, spiece_model))
        tokenizer_layer = T5TokenizerLayer(
            lower_case=False,
            model_file_path=spiece_model,
            out_type=out_type,
            strip_diacritics=True,
            sep_token_id=tokenizer.sep_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=None,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            cls_enc_token_id=32000,
            cls_dec_token_id=32002,
            mask_token_id=32001,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            pack_model_inputs=pack_model_inputs,
            dynamic_padding=dynamic_padding,
            truncate=truncate,
        )
        return tokenizer_layer
