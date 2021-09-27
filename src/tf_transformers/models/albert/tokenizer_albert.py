#!/usr/bin/env python
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
"""Albert Tokenizer based on TFText"""
import tempfile
from pathlib import Path
from typing import Dict, Union

import sentencepiece
import tensorflow as tf
import tensorflow_text as tf_text
from absl import logging

_PREFIX_DIR = 'tftransformers_tokenizer_cache'


def get_vocab(model_proto):
    """Get vocab from sentencpiece model"""
    sp_model = sentencepiece.SentencePieceProcessor()
    sp_model.LoadFromSerializedProto(model_proto)
    vocab = {sp_model.IdToPiece(i): i for i in range(sp_model.GetPieceSize())}
    return vocab


class AlbertTokenizerLayer(tf.keras.layers.Layer):
    """Wraps `tf_text.SentencepieceTokenizer` as a Keras Layer.
    Attributes:
    tokenize_with_offsets: If true, calls
      `SentencepieceTokenizer.tokenize_with_offsets()`
      instead of plain `.tokenize()` and outputs a triple of
      `(tokens, start_offsets, limit_offsets)`.
    """

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
        cls_token_id=None,
        sep_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        unk_token_id=None,
        pad_token_id=None,
        max_length=None,
        add_special_tokens=False,
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
        Raises:
          ImportError: if importing tensorflow_text failed.
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
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id

        self.max_length = max_length

        self.add_special_tokens = add_special_tokens

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
        batch_size = tf.shape(inputs)[0]

        tokens = self._tokenizer.tokenize(inputs)
        # If Truncation is True
        if truncation:
            if max_length:
                tokens = tokens[:, :max_length]
            else:
                if add_special_tokens:
                    tokens = tokens[:, self.max_length - 2]  # For cls and sep
                else:
                    tokens = tokens[:, self.max_length]
        # If add special_tokens
        if add_special_tokens:
            tokens = self._add_special_tokens(tokens)

        # If padding
        if padding:
            if max_pad_length is None:
                raise ValueError("If padding is True, please set max_pad_length")
            if isinstance(tokens, tf.RaggedTensor):
                tokens = tokens.to_tensor(default_value=self.pad_token_id, shape=(batch_size, max_pad_length))
        # If return_tensors='tf'
        if return_tensors == 'tf':
            if isinstance(tokens, tf.RaggedTensor):
                tokens = tokens.to_tensor(default_value=self.pad_token_id)

        result = {}
        result['input_ids'] = tokens
        result['input_mask'] = tf.ones_like(tokens)
        result['input_type_ids'] = tf.zeros_like(tokens)

        return result

    def encode_deprecated(
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
        batch_size = tf.shape(inputs)[0]

        def _reshape(rt):
            values = rt.values
            row_splits = rt.row_splits
            row_splits = tf.reshape(row_splits, [batch_size + 1])
            return tf.RaggedTensor.from_row_splits(values, row_splits)

        # Call the tokenizer.
        if self.tokenize_with_offsets:
            tokens, start_offsets, limit_offsets = self._tokenizer.tokenize_with_offsets(inputs)

            tokens = _reshape(tokens)
            start_offsets = _reshape(start_offsets)
            limit_offsets = _reshape(limit_offsets)

            return {'input_ids': tokens, 'start_offsets': start_offsets, 'limit_offsets': limit_offsets}
        else:
            tokens = self._tokenizer.tokenize(inputs)
            tokens = _reshape(tokens)

            # If Truncation is True
            if truncation:
                if max_length:
                    tokens = tokens[:, :max_length]
                else:
                    if add_special_tokens:
                        tokens = tokens[:, self.max_length - 2]  # For cls and sep
                    else:
                        tokens = tokens[:, self.max_length]
            # If add special_tokens
            if add_special_tokens:
                tokens = self._add_special_tokens(tokens)

            # If padding
            if padding:
                if max_pad_length is None:
                    raise ValueError("If padding is True, please set max_pad_length")
                if isinstance(tokens, tf.RaggedTensor):
                    tokens = tokens.to_tensor(default_value=self.pad_token_id, shape=(batch_size, max_pad_length))
            # If return_tensors='tf'
            if return_tensors == 'tf':
                if isinstance(tokens, tf.RaggedTensor):
                    tokens = tokens.to_tensor(default_value=self.pad_token_id)

            return {'input_ids': tokens}

    def call_encode(self, inputs: Dict[str, tf.Tensor]):
        """Encode function used for serialization"""
        inputs = tf.squeeze(inputs['text'], axis=0)
        tokens = self._tokenizer.tokenize(inputs)
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

        # If add special_tokens
        if self.add_special_tokens:
            tokens = self._add_special_tokens(tokens)

        result = {}
        result['input_ids'] = tokens
        result['input_mask'] = tf.ones_like(tokens)
        result['input_type_ids'] = tf.zeros_like(tokens)

        return result

    def decode(self, inputs: Union[tf.Tensor, tf.RaggedTensor]):
        """Encode function used for serialization"""
        decoded_tokens = self._tokenizer.detokenize(inputs)
        return decoded_tokens

    def call(self, inputs):
        results = self.call_encode(inputs)
        return results

    def get_config(self):
        # Skip in tf.saved_model.save(); fail if called direcly.
        # raise NotImplementedError("TODO(b/170480226): implement")
        pass

    def _add_special_tokens(self, tokens):
        return tf_text.combine_segments(
            [tokens], start_of_sequence_id=self.cls_token_id, end_of_segment_id=self.sep_token_id
        )[0]

    def get_model(self):
        """Convert Keras Layer to Model"""
        from tf_transformers.core import LegacyModel

        inputs = {
            "text": tf.keras.layers.Input(
                shape=(None,),
                batch_size=1,
                dtype=tf.string,
                name="text",
            )
        }
        layer_outputs = self(inputs)
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name='albert_sentencepice')
        return model


def create_cache_dir(cache_path: Path):
    """Create Cache Directory

    Args:
        cache_path : Path
    """
    if not cache_path.exists():  # If cache path not exists
        cache_path.mkdir()


class AlbertTokenizerTFText:
    def __init__(self) -> None:
        pass

    @classmethod
    def from_pretrained(cls, model_name: str, out_type=tf.int32):
        """We load tokenizer from HuggingFace and pass to TFtext"""
        cache_dir = tempfile.gettempdir()
        cache_dir = Path(cache_dir, _PREFIX_DIR)
        create_cache_dir(cache_dir)

        cache_path = Path(cache_dir, model_name)

        from transformers import AlbertTokenizer

        tokenizer = AlbertTokenizer.from_pretrained(model_name)
        if not cache_path.exists():
            tokenizer.save_pretrained(str(cache_path))
            logging.info("Saving {} tokenizer to {}".format(model_name, cache_path))
        spiece_model = str(Path(cache_path, 'spiece.model'))
        logging.info("Loading {} tokenizer to {}".format(model_name, spiece_model))
        tokenizer_layer = AlbertTokenizerLayer(
            lower_case=True,
            model_file_path=spiece_model,
            out_type=out_type,
            strip_diacritics=False,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=None,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=tokenizer.max_len_single_sentence,
        )
        return tokenizer_layer
