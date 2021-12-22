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
    def text_to_features_batch(item):
        """Convert text to features by packing maximum contiguos sequences to ids"""
        # Squeeze item
        item = {'text': tf.squeeze(item['text'], axis=1)}
        # Encode
        input_ids = tokenizer_layer(item)
        # Ragged to tensor
        input_ids = input_ids.merge_dims(-2, 1)
        # Add cls_token_id and eos_id to the end
        input_ids = tf.concat([input_ids[: encoder_seq_length - 2], [cls_token_id], [eos_id]], axis=0)
        input_ids = tf.RaggedTensor.from_tensor(tf.expand_dims(input_ids, axis=0))
        return {'input_ids': input_ids}

    def mask_and_prepare_inputs(item):

        input_ids = item['input_ids'].merge_dims(1, 2)
        # Apply dynamic masking, with expand_dims on the input batch
        # If expand_dims is not there, whole word masking fails
        masked_token_ids, masked_pos, masked_lm_ids = tf_text.mask_language_model(
            input_ids,
            item_selector=random_selector,
            mask_values_chooser=mask_values_chooser,
        )

        # Prepare and pad combined segment inputs
        input_word_ids, input_mask = tf_text.pad_model_inputs(masked_token_ids, max_seq_length=encoder_seq_length)

        # Original unmasked ids
        input_original_ids, _ = tf_text.pad_model_inputs(input_ids, max_seq_length=encoder_seq_length)

        # Decoder inputs
        decoder_start_ids = tf.ones((batch_size, 1), dtype=tf.int32) * decoder_start_token_id
        decoder_input_ids = tf.concat([decoder_start_ids, input_original_ids], axis=1)

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
        inputs['encoder_input_ids'] = input_word_ids
        inputs['encoder_input_mask'] = input_mask
        inputs['decoder_input_ids'] = decoder_input_ids

        inputs['masked_lm_positions'] = masked_lm_positions

        outputs = {}
        outputs['labels'] = labels
        outputs['labels_mask'] = labels_mask
        outputs['masked_lm_labels'] = masked_lm_ids
        outputs['masked_lm_weights'] = masked_lm_weights

        return inputs, outputs

    dataset = read_dataset(data_directory)

    local_batch = 7  # This is used to pack maximum input tokens per sentences/example
    encoder_seq_length = max_seq_len

    eos_id = tokenizer_layer.eos_token_id
    cls_token_id = tokenizer_layer.cls_token_id
    decoder_start_token_id = tokenizer_layer.pad_token_id

    # Random Selector (10 per)

    max_predictions_per_seq = 100
    selection_rate = 0.20
    unk_token_id = tokenizer_layer.unk_token_id
    pad_token_id = tokenizer_layer.pad_token_id
    eos_token_id = tokenizer_layer.eos_token_id

    random_selector = tf_text.RandomItemSelector(
        max_selections_per_batch=max_predictions_per_seq,
        selection_rate=selection_rate,
        unselectable_ids=[unk_token_id, eos_token_id, pad_token_id, cls_token_id],
    )

    # Mask Value chooser (Encapsulates the BERT MLM token selection logic)
    vocab_size = tokenizer_layer.vocab_size
    mask_token_id = tokenizer_layer.mask_token_id
    mask_values_chooser = tf_text.MaskValuesChooser(vocab_size, mask_token_id, 0.8)

    ds = dataset.batch(local_batch)
    ds = ds.map(text_to_features_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(mask_and_prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
