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

import tensorflow as tf


def get_dataset(data_directory, tokenizer_layer, max_seq_len, batch_size):
    """Convert text to tf.data.Dataset after map fn

    Args:
        data_directory (:obj:`str`): Text data directory. For TPU, it will be in GCP bucket.
        tokenizer_layer (:obj:`tf_transformers.core.LegacyLayer`): Tokenizer layer based on tf text.
        max_seq_len (:obj:`int`): Maximum sequence length for each example.
        batch_size (:obj:`int`): Batch size for tf.data.Dataset

    Returns:
        A function to be used in tf.data.Dataset.map
    """

    def text_to_featutes(src_text, target_text):
        """Convert item a tuple (src, target) into features"""
        src_input_ids = tokenizer_layer({'text': [src_text]})
        target_input_ids = tokenizer_layer({'text': [target_text]})

        # We will remove [1] EOS_ID from input_ids for target
        # and add DECODER_START_TOKEN_ID [0] in the start
        target_input_ids = target_input_ids[:, :-1]
        target_input_ids = tf.concat([[[0]], target_input_ids], axis=1)

        src_input_ids = tf.squeeze(src_input_ids, axis=0)
        target_input_ids = tf.squeeze(target_input_ids, axis=0)
        return {
            'encoder_input_ids': src_input_ids,
            'encoder_input_mask': tf.ones_like(src_input_ids),
            'decoder_input_ids': target_input_ids,
        }

    def filter_by_length(item, max_seq_len):
        """When an example doesn't have multiple sentences\
            there wont be any masked sentence. Ignore those examples,
            as nothing to predict.
            """

        # Both encoder and decoder has to be inside the specified max_seq_len
        # max_seq_len + 1 is for decoder_input_ids ( as we shift it one position for labels)
        if tf.less_equal(tf.shape(item['encoder_input_ids'])[0], max_seq_len) and tf.less_equal(
            tf.shape(item['decoder_input_ids'])[0], max_seq_len + 1
        ):
            return tf.constant(True)
        else:
            return tf.constant(False)

    def separate_x_y(item):
        """Separate x and y for batch_inputs and batch_labels"""
        x = {}
        y = {}

        x['encoder_input_ids'] = item['encoder_input_ids']
        x['encoder_input_mask'] = item['encoder_input_mask']
        x['decoder_input_ids'] = item['decoder_input_ids'][:, :-1]  # Shift one position .

        y['labels'] = item['decoder_input_ids'][:, 1:]
        y['labels_mask'] = tf.ones_like(y['labels'])

        return x, y

    c4_files = tf.data.Dataset.list_files(os.path.join(data_directory, "*.tsv*"))
    c4_lines = c4_files.interleave(lambda fname: tf.data.TextLineDataset(fname).skip(1), cycle_length=100)

    # Parse tsv
    c4_column_types = [str(), str()]
    ds = c4_lines.map(
        lambda x: tf.io.decode_csv(x, field_delim='\t', record_defaults=c4_column_types, use_quote_delim=False)
    )
    ds = ds.map(text_to_featutes, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter out those examples which are greater than max_seq_len
    ds = ds.filter(lambda x: filter_by_length(x, max_seq_len=max_seq_len))

    # Shuffle and Prefetch
    ds = ds.shuffle(1024, reshuffle_each_iteration=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Batch to fixed shapes, TPU requires that
    _padded_shapes = {
        'encoder_input_ids': [max_seq_len],
        'encoder_input_mask': [max_seq_len],
        'decoder_input_ids': [max_seq_len + 1],
    }

    ds = ds.padded_batch(batch_size, padded_shapes=_padded_shapes, drop_remainder=True)

    # Auto SHARD
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    ds = ds.with_options(options)

    # Ignore any erros
    ds = ds.apply(tf.data.experimental.ignore_errors(log_warning=True))

    # Separate to inputs and labels

    ds = ds.map(separate_x_y, num_parallel_calls=tf.data.AUTOTUNE)

    return ds
