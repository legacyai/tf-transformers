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


def prefix_lm_fn(tokenizer_layer, max_seq_len, add_eos_after_prefix=True):
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
        # Find the sentence boundary which is less than or equal to max_seq_len (-3 for CLS, EOS, SEP)
        if add_eos_after_prefix:
            buffer_length = 3
        else:
            buffer_length = 2
        max_seq_index = tf.where(input_ids.row_splits < max_seq_len - buffer_length)[-1][0]
        # Trim the inputs
        input_ids = input_ids[:max_seq_index]
        # We take random position between 1 and len(sentences)//2
        mid_index = (tf.shape(input_ids.row_splits)[0] - 1) // 2

        # Sometime, 2nd sentence might go beyond max_seq_len
        # Then we will have only 1, None shape for input_ids
        # Then 1//2 == 0 for mid_index fails. In those case, we randomly chunk the sentence
        # without looking for a proper sentence end.
        # This will result in lesser sequences for some examples
        if tf.equal(mid_index, 0):
            max_seq_index = input_ids.row_splits[-1]
            mid_index = tf.cast(max_seq_index // 2, tf.int32)
            prefix_mask_index = tf.random.uniform(minval=0, maxval=mid_index + 1, shape=(), dtype=tf.int32)
            input_ids_first_portion = input_ids[:, :prefix_mask_index]
            input_ids_second_portion = input_ids[:, prefix_mask_index:]
        else:
            prefix_mask_index = tf.random.uniform(minval=1, maxval=mid_index + 1, shape=(), dtype=tf.int32)
            # We split it to 2 parts left and right
            # left we mask by 1 and right we mask by 0
            # right side portions are our targets
            input_ids_first_portion = input_ids[:prefix_mask_index]
            input_ids_second_portion = input_ids[prefix_mask_index:]
        # We use CLS token id as EOS for prefix models. The reason is, otherwise
        # model might not predict next sentence, as it encounter . or delimiter much frequently during end of
        # example.
        if add_eos_after_prefix:
            input_ids_first_portion = tf.concat([input_ids_first_portion, [[cls_token_id]]], axis=0)
        # Shift 1 position right to account for EOS token (CLS here)
        prefix_mask_index = prefix_mask_index + 1

        # Split and join
        input_mask_first_portion = tf.ones_like(input_ids_first_portion)
        input_mask_second_portion = tf.zeros_like(input_ids_second_portion)
        input_mask = tf.concat([input_mask_first_portion, input_mask_second_portion], axis=0)
        input_ids = tf.concat([input_ids_first_portion, input_ids_second_portion], axis=0)

        # Pad inputs
        input_ids_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(input_ids.merge_dims(-2, 1), 0))
        input_mask_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(input_mask.merge_dims(-2, 1), 0))

        # Add CLS and SEP
        input_ids_ragged = tf.concat([[[cls_token_id]], input_ids_ragged, [[sep_token_id]]], axis=1)
        # 0 here is for extra SEP which has to be predicted as EOS token while generation
        input_mask_ragged = tf.concat([[[1]], input_mask_ragged, [[0]]], axis=1)

        # Opposite of input_mask (Do it before input_mask padding)
        lm_label_weights = tf.cast(tf.not_equal(input_mask_ragged, 1), tf.int32)
        input_word_ids, _ = tf_text.pad_model_inputs(input_ids_ragged, max_seq_length=max_seq_len + 1)
        input_mask, _ = tf_text.pad_model_inputs(input_mask_ragged, max_seq_length=max_seq_len + 1)
        lm_label_weights, _ = tf_text.pad_model_inputs(lm_label_weights, max_seq_length=max_seq_len + 1)

        # Squeeze here will help to retain 2D when we batch outside map fn
        input_word_ids = tf.squeeze(input_word_ids, axis=0)
        input_mask = tf.squeeze(input_mask, axis=0)
        lm_label_weights = tf.squeeze(lm_label_weights, axis=0)

        # Shift positions
        lm_labels = input_word_ids[1:]
        input_word_ids = input_word_ids[:-1]
        input_mask = input_mask[:-1]
        lm_label_weights = lm_label_weights[1:]

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
