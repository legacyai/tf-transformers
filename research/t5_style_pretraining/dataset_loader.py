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

import os
from random import shuffle

import tensorflow as tf
import tensorflow_text as tf_text

from tf_transformers.data import TFReader


def read_dataset(data_directory):
    schema = {"text": ("var_len", "bytes")}
    all_files = tf.io.gfile.glob(os.path.join(data_directory, "c4/en/3.0.1/*.tfrecord*"))
    shuffle(all_files)
    tf_reader = TFReader(schema=schema, tfrecord_files=all_files)
    dataset = tf_reader.read_record(keys=['text'])
    return dataset


def get_dataset(data_directory, tokenizer_layer, max_seq_len, batch_size):
    def mask_and_prepare_inputs(item):

        # text = tf.strings.join(tf.concat(tf.squeeze(item['text'], axis=1), axis=0))
        text = tf.strings.reduce_join([tf.squeeze(item['text'], axis=1)])
        # Encode
        inputs = {'text': tf.strings.split(text, delimiter)}
        segments = tokenizer_layer(inputs)

        # Find the index which is equal to or greater (next greater) max_seq_len
        max_seq_index = tf.where(segments.row_splits >= encoder_seq_length - 3)[0][
            0
        ]  # -3 to accomodate CLS_ENC, CLS_DEC, EOS
        # Trim based on max_seq_len, As its a ragged tensor, we find max_seq_index
        segments = segments[:max_seq_index]

        if tf.not_equal(tf.shape(segments.merge_dims(-2, 1))[0], encoder_seq_length - 3):
            difference = tf.shape(segments.merge_dims(-2, 1))[0] - (encoder_seq_length - 3)
            difference = tf.cast(difference, tf.int64)
            row_splits = segments.row_splits
            row_splits = tf.concat([row_splits[:-1], [row_splits[-1] - difference]], axis=0)
            segments = tf.concat([segments[:-1], [segments[-1][:-difference]]], axis=0)

        # Add 3 special tokens
        segments = tf.concat([[[cls_enc_token_id]], segments, [[cls_dec_token_id]], [[eos_token_id]]], axis=0)

        # Apply dynamic masking, with expand_dims on the input batch
        # If expand_dims is not there, whole word masking fails
        masked_token_ids, masked_pos, masked_lm_ids = tf_text.mask_language_model(
            tf.expand_dims(segments, axis=0),  # super important
            item_selector=random_selector,
            mask_values_chooser=mask_values_chooser,
        )

        # Prepare and pad combined segment inputs
        input_word_ids, input_mask = tf_text.pad_model_inputs(masked_token_ids, max_seq_length=encoder_seq_length)

        # Original unmasked ids
        input_ids = tf.RaggedTensor.from_tensor(tf.expand_dims(segments.merge_dims(-2, 1), 0))
        input_original_ids, _ = tf_text.pad_model_inputs(input_ids, max_seq_length=encoder_seq_length)

        # Decoder inputs
        decoder_input_ids = tf.concat([[[decoder_start_token_id]], input_original_ids], axis=1)

        labels = decoder_input_ids[:, 1:]  # Shift 1 position right
        # Make non pad_token_id positions 1 and pad_token_id pos 0
        labels_mask = tf.cast(tf.not_equal(labels, pad_token_id), tf.int32)

        decoder_input_ids = decoder_input_ids[:, :-1]  # Except last word

        # Prepare and pad masking task inputs
        # Masked lm weights will mask the weights
        masked_lm_positions, masked_lm_weights = tf_text.pad_model_inputs(
            masked_pos, max_seq_length=max_predictions_per_seq
        )
        masked_lm_ids, _ = tf_text.pad_model_inputs(masked_lm_ids, max_seq_length=max_predictions_per_seq)

        inputs = {}
        # inputs['encoder_input_ids'] = tf.squeeze(input_word_ids, axis=0)
        # inputs['encoder_input_mask'] = tf.squeeze(input_mask, axis=0)
        # inputs['decoder_input_ids'] = tf.squeeze(decoder_input_ids, axis=0)

        inputs['input_ids'] = tf.squeeze(input_word_ids, axis=0)
        inputs['input_mask'] = tf.squeeze(input_mask, axis=0)


        inputs['masked_lm_positions'] = tf.squeeze(masked_lm_positions, axis=0)

        outputs = {}
        outputs['labels'] = tf.squeeze(labels, axis=0)
        outputs['labels_mask'] = tf.squeeze(labels_mask, axis=0)
        outputs['masked_lm_labels'] = tf.squeeze(masked_lm_ids, axis=0)
        outputs['masked_lm_weights'] = tf.squeeze(masked_lm_weights, axis=0)

        return inputs, outputs

    dataset = read_dataset(data_directory)

    local_batch = 10  # This is used to pack maximum input tokens per sentences/example
    encoder_seq_length = max_seq_len

    cls_enc_token_id = tokenizer_layer.cls_enc_token_id
    cls_dec_token_id = tokenizer_layer.cls_dec_token_id
    decoder_start_token_id = tokenizer_layer.pad_token_id
    delimiter = ' '

    # Random Selector (10 per)

    max_predictions_per_seq = 100
    selection_rate = 0.20
    unk_token_id = tokenizer_layer.unk_token_id
    pad_token_id = tokenizer_layer.pad_token_id
    eos_token_id = tokenizer_layer.eos_token_id

    random_selector = tf_text.RandomItemSelector(
        max_selections_per_batch=max_predictions_per_seq,
        selection_rate=selection_rate,
        unselectable_ids=[unk_token_id, eos_token_id, pad_token_id, cls_enc_token_id, cls_dec_token_id],
    )

    # Mask Value chooser (Encapsulates the BERT MLM token selection logic)
    vocab_size = tokenizer_layer.vocab_size
    mask_token_id = tokenizer_layer.mask_token_id
    mask_values_chooser = tf_text.MaskValuesChooser(vocab_size, mask_token_id, 0.8)

    ds = dataset.batch(local_batch)
    ds = ds.map(mask_and_prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds
