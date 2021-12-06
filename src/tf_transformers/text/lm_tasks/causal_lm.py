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
"""The main wrapper around Causal Language Model Tasks"""
import tensorflow as tf
import tensorflow_text as tf_text


def causal_lm_fn(tokenizer_layer, max_seq_len, add_cls_token=True):
    """The main function for CLM.

    Args:
        tokenizer_layer : A tokenizer layer from tf_transformers. eg: AlbertTokenizerTFText
        max_seq_len (:obj:`int`): Max sequence length of input
        add_cls_sep (:obj:`bool`): Whether to add CLS and SEP token

    Returns:
        A function, which can be used with tf.data.Dataset.map
    """
    cls_token_id = tokenizer_layer.cls_token_id
    sep_token_id = tokenizer_layer.sep_token_id
    unk_token_id = tokenizer_layer.unk_token_id  # noqa
    pad_token_id = tokenizer_layer.pad_token_id  # noqa
    mask_token_id = tokenizer_layer.mask_token_id  # noqa
    vocab_size = tokenizer_layer.vocab_size  # noqa

    def causal_map_fn(item):
        # We expect item to be dict of {'text': ['sentence1']}
        input_ids = tokenizer_layer(item)
        if add_cls_token:
            # Trim inputs (+1 is because for Causal LM we shift inputs and labels)
            input_ids_ragged = input_ids[:, : max_seq_len + 1 - 2]
            input_ids_ragged = tf.concat([[[cls_token_id]], input_ids_ragged, [[sep_token_id]]], axis=1)
        else:
            # Trim inputs (+1 is because for Causal LM we shift inputs and labels)
            input_ids_ragged = input_ids[:, : max_seq_len + 1 - 1]
            input_ids_ragged = tf.concat([input_ids_ragged, [[sep_token_id]]], axis=1)

        # input_mask = tf.ones_like(input_ids_ragged)
        lm_label_weights = tf.ones_like(input_ids_ragged)
        input_word_ids, _ = tf_text.pad_model_inputs(input_ids_ragged, max_seq_length=max_seq_len + 1)
        # input_mask, _ = tf_text.pad_model_inputs(input_mask, max_seq_length=max_seq_len + 1)
        lm_label_weights, _ = tf_text.pad_model_inputs(lm_label_weights, max_seq_length=max_seq_len + 1)

        # Squeeze here will help to retain 2D when we batch outside map fn
        input_word_ids = tf.squeeze(input_word_ids, axis=0)
        # input_mask = tf.squeeze(input_mask, axis=0)
        lm_label_weights = tf.squeeze(lm_label_weights, axis=0)

        # Shift positions
        lm_labels = input_word_ids[1:]
        input_word_ids = input_word_ids[:-1]
        # input_mask = input_mask[:-1]
        lm_label_weights = lm_label_weights[1:]  # We attend all labels, as we are Causal LM

        inputs = {}
        inputs['input_ids'] = input_word_ids
        inputs['input_type_ids'] = tf.zeros_like(input_word_ids)

        labels = {}
        labels['lm_labels'] = lm_labels
        labels['lm_weights'] = lm_label_weights

        return (inputs, labels)

    return causal_map_fn
