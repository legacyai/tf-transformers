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
"""The main wrapper around Masked Language Model Tasks"""
import tensorflow as tf
import tensorflow_text as tf_text


def mlm_fn(tokenizer_layer, max_seq_len, max_predictions_per_seq, delimiter=' '):
    """The main function for MLM.

    Args:
        tokenizer_layer : A tokenizer layer from tf_transformers. eg: AlbertTokenizerTFText
        max_seq_len (:obj:`int`): Max sequence length of input
        max_predictions_per_seq (:obj:`int`): Maximum predictions (Masked tokens) per sequence
        delimiter (:obj:`str`): A delimiter with which we split. Default is whitespace.

    Returns:
        A function, which can be used with tf.data.Dataset.map
    """
    cls_token_id = tokenizer_layer.cls_token_id
    sep_token_id = tokenizer_layer.sep_token_id
    unk_token_id = tokenizer_layer.unk_token_id
    pad_token_id = tokenizer_layer.pad_token_id
    mask_token_id = tokenizer_layer.mask_token_id
    vocab_size = tokenizer_layer.vocab_size

    # Random Selector (10 per)
    random_selector = tf_text.RandomItemSelector(
        max_selections_per_batch=max_predictions_per_seq,
        selection_rate=0.1,
        unselectable_ids=[cls_token_id, sep_token_id, unk_token_id, pad_token_id],
    )

    # Mask Value chooser (Encapsulates the BERT MLM token selection logic)
    mask_values_chooser = tf_text.MaskValuesChooser(vocab_size, mask_token_id, 0.8)

    def dynamic_mlm(example):
        # We expect a dict example
        text = example['text']
        # TODO : This might not be required .
        if tokenizer_layer._lower_case:
            text = tf_text.case_fold_utf8(text)

        inputs = {'text': tf.strings.split(text, delimiter)}
        segments = tokenizer_layer(inputs)
        # Find the index where max_seq_len is valid
        max_seq_index = tf.where(segments.row_splits < max_seq_len - 2)[-1][0]
        # Trim based on max_seq_len, As its a ragged tensor, we find max_seq_index
        segments = segments[:max_seq_index]

        # Randomize slice inorder to avoid bias
        # 0.05 per time, we slice the inputs between 3 words and len(segments)
        # 50 % time we slice from the begining , 50 % we slice from last
        slice_prob = tf.random.uniform(shape=())
        if slice_prob <= 0.05:
            MIN_WORDS = 3
            max_segment_shape = tf.shape(segments.row_splits)[0] - 1
            # minval = tf.minimum(MIN_WORDS, max_segment_shape)
            # maxval = tf.maximum(MIN_WORDS, max_segment_shape)
            if tf.greater(max_segment_shape, MIN_WORDS):
                random_slice_index = tf.random.uniform(
                    minval=MIN_WORDS, maxval=max_segment_shape, shape=(), dtype=tf.int32
                )
                right_left_prob = tf.random.uniform(shape=())
                if right_left_prob <= 0.5:
                    segments = segments[:random_slice_index]
                else:
                    segments = segments[random_slice_index:]

        # Flatten and add CLS , SEP
        segments_flattened = segments.merge_dims(-2, 1)
        segments_combined = tf.concat([[cls_token_id], segments_flattened, [sep_token_id]], axis=0)
        # We have to move original row splits to acoomoadate 2 extra tokens added later, CLS and SEP
        row_splits = tf.concat([[0], segments.row_splits + 1, [segments.row_splits[-1] + 2]], axis=0)
        segments_combined = tf.RaggedTensor.from_row_splits(segments_combined, row_splits)
        # Apply dynamic masking, with expand_dims on the input batch
        # If expand_dims is not there, whole word masking fails
        masked_token_ids, masked_pos, masked_lm_ids = tf_text.mask_language_model(
            tf.expand_dims(segments_combined, axis=0),
            item_selector=random_selector,
            mask_values_chooser=mask_values_chooser,
        )

        # Prepare and pad combined segment inputs
        input_word_ids, input_mask = tf_text.pad_model_inputs(masked_token_ids, max_seq_length=max_seq_len)
        input_type_ids = tf.zeros_like(input_word_ids)

        # Prepare and pad masking task inputs
        # Masked lm weights will mask the weights
        masked_lm_positions, masked_lm_weights = tf_text.pad_model_inputs(
            masked_pos, max_seq_length=max_predictions_per_seq
        )
        masked_lm_ids, _ = tf_text.pad_model_inputs(masked_lm_ids, max_seq_length=max_predictions_per_seq)

        # Work around broken shape inference.
        output_shape = tf.stack(
            [masked_token_ids.nrows(out_type=tf.int32), tf.cast(max_seq_len, dtype=tf.int32)]  # batch_size
        )
        output_shape_masked_tokens = tf.stack(
            [masked_pos.nrows(out_type=tf.int32), tf.cast(max_predictions_per_seq, dtype=tf.int32)]  # batch_size
        )

        def _reshape(t):
            return tf.reshape(t, output_shape)

        def _reshape_masked(t):
            return tf.reshape(t, output_shape_masked_tokens)

        input_word_ids = _reshape(input_word_ids)
        input_type_ids = _reshape(input_type_ids)
        input_mask = _reshape(input_mask)

        masked_lm_positions = _reshape_masked(masked_lm_positions)
        masked_lm_ids = _reshape_masked(masked_lm_ids)
        masked_lm_weights = _reshape_masked(masked_lm_weights)

        inputs = {}
        inputs['input_ids'] = tf.squeeze(input_word_ids, axis=0)
        inputs['input_type_ids'] = tf.squeeze(input_type_ids, axis=0)
        inputs['input_mask'] = tf.squeeze(input_mask, axis=0)
        inputs['masked_lm_positions'] = tf.squeeze(masked_lm_positions, axis=0)

        labels = {}
        labels['masked_lm_labels'] = tf.squeeze(masked_lm_ids, axis=0)
        labels['masked_lm_weights'] = tf.squeeze(masked_lm_weights, axis=0)  # Mask

        return (inputs, labels)

    return dynamic_mlm
