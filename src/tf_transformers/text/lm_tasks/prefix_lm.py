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
"""The main wrapper around Prefix Language Model Tasks"""
import tensorflow as tf
import tensorflow_text as tf_text


def prefix_lm_fn(tokenizer_layer, max_seq_len, add_cls_sep=False):
    """The main function for PLM.

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

    def prefix_map_fn(item):
        input_ids = tokenizer_layer({'text': item['sentences']})
        # We take random position between 1 and len(sentences)//2
        mid_index = tf.shape(input_ids.flat_values)[0] // 2
        prefix_mask_index = tf.random.uniform(minval=1, maxval=mid_index + 1, shape=(), dtype=tf.int32)
        # We split it to 2 parts left and right
        # left we mask by 1 and right we mask by 0
        # right side portions are our targets
        input_ids_first_portion = input_ids[:prefix_mask_index]
        input_ids_second_portion = input_ids[prefix_mask_index:]

        # Split and join
        input_mask_first_portion = tf.ones_like(input_ids_first_portion)
        input_mask_second_portion = tf.zeros_like(input_ids_second_portion)
        input_mask = tf.concat([input_mask_first_portion, input_mask_second_portion], axis=0)
        # Pad inputs
        input_ids_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(input_ids.merge_dims(-2, 1), 0))
        input_mask_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(input_mask.merge_dims(-2, 1), 0))
        # Trim inputs (+1 is because for Causal LM we shift inputs and labels)
        if add_cls_sep:
            input_ids_ragged = input_ids_ragged[:, : max_seq_len + 1 - 2]
            input_mask_ragged = input_mask_ragged[:, : max_seq_len + 1 - 2]

            input_ids_ragged = tf.concat([[[cls_token_id]], input_ids_ragged, [[sep_token_id]]], axis=1)
            input_mask_ragged = tf.concat([[[1]], input_mask_ragged, [[1]]], axis=1)

        else:
            input_ids_ragged = input_ids_ragged[:, : max_seq_len + 1]
            input_mask_ragged = input_mask_ragged[:, : max_seq_len + 1]

        input_word_ids, _ = tf_text.pad_model_inputs(input_ids_ragged, max_seq_length=max_seq_len + 1)
        input_mask, _ = tf_text.pad_model_inputs(input_mask_ragged, max_seq_length=max_seq_len + 1)

        # Squeeze here will help to retain 2D when we batch outside map fn
        input_word_ids = tf.squeeze(input_word_ids, axis=0)
        input_mask = tf.squeeze(input_mask, axis=0)

        # Shift positions
        lm_labels = input_word_ids[1:]
        input_word_ids = input_word_ids[:-1]
        input_mask = input_mask[:-1]
        # Opposite of input_mask
        lm_label_weights = tf.cast(tf.not_equal(input_mask, 1), tf.int32)

        inputs = {}
        inputs['input_ids'] = input_word_ids
        inputs['input_mask'] = input_mask
        inputs['input_type_ids'] = tf.zeros_like(input_word_ids)

        labels = {}
        labels['lm_labels'] = lm_labels
        labels['lm_weights'] = lm_label_weights
        labels['prefix_mask_index'] = prefix_mask_index

        return (inputs, labels)

    return prefix_map_fn