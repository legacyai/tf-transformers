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

from tf_transformers.layers.mask import prefix_mask
from tf_transformers.layers.mask.causal_mask import attention_mask_square
from tf_transformers.text.lm_tasks import causal_lm_fn, mlm_fn, prefix_lm_fn


def get_dataset(data_directory, tokenizer_layer, max_seq_len, batch_size, minimum_prefix_length=900):
    """Convert text to tf.data.Dataset after map fn

    Args:
        data_directory (:obj:`str`): Text data directory. For TPU, it will be in GCP bucket.
        tokenizer_layer (:obj:`tf_transformers.core.LegacyLayer`): Tokenizer layer based on tf text.
        max_seq_len (:obj:`int`): Maximum sequence length for each example.
        batch_size (:obj:`int`): Batch size for tf.data.Dataset
        minimum_prefix_length (:obj:`int`): Minimum length required for prefix LM.

    Returns:
        A function to be used in tf.data.Dataset.map
    """

    prefix_map_fn = prefix_lm_fn(tokenizer_layer, max_seq_len)
    masked_lm_map_fn = mlm_fn(tokenizer_layer, max_seq_len, max_seq_len)
    causal_lm_map_fn = causal_lm_fn(tokenizer_layer, max_seq_len)

    def split_text(item):
        """Split text into list of sentences"""
        sentences = tf.strings.split(item, '__||__')
        return {'sentences': sentences}

    def filter_empty_string(item):
        """This will ensure, if any of the sentence in list of sentences is '' or' ', empty string,
        that will be filtered out"""
        sentences = item['sentences']
        valid_string_indexes = tf.squeeze(tf.where(tf.not_equal(tf.strings.length(item['sentences']), 0)), axis=1)
        sentences = tf.gather(sentences, valid_string_indexes)
        item['sentences'] = sentences
        return item

    # def filter_single_sentence(item):
    #     """If number of sentences after split is 1, ignore, because nothing to prefix, as we have only one sentence"""
    #     number_of_sentences = tf.shape(item['sentences'])[0]
    #     return tf.greater(number_of_sentences, tf.constant(1))

    def filter_out_empty_mask(x, y):
        """When an example doesn't have multiple sentences\
            there wont be any masked sentence. Ignore those examples,
            as nothing to predict.
            """
        return tf.greater(tf.reduce_sum(tf.cast(tf.not_equal(y['masked_lm_weights'], 0), tf.int32)), 0)

    def prepare_3d_input_mask_mlm(input_mask):
        """Prepare 3D mask from 2D"""
        batch_size = tf.shape(input_mask)[0]
        seq_length = tf.shape(input_mask)[1]

        to_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, seq_length]), dtype=input_mask.dtype)
        broadcast_ones = tf.ones(shape=[batch_size, seq_length, 1], dtype=input_mask.dtype)

        mask = broadcast_ones * to_mask

        return tf.squeeze(tf.cast(mask, tf.float32), axis=0)

    def filter_prefix_with_minimum_length(x, y):
        """Sometime prefix mask have smaller sequences, ignore that"""
        min_sequence_length = minimum_prefix_length
        # Check sequence length by count padding tokens
        non_padded_count = tf.reduce_sum(tf.cast(tf.not_equal(x['input_ids'], 0), tf.int32))
        if tf.equal(y['type'], b'prefix'):
            if tf.greater_equal(non_padded_count, min_sequence_length):
                return tf.constant(True)
            else:
                return tf.constant(False)
        # If not prefix let it pass
        else:
            return tf.constant(True)

    def rename_labels_dict(x, y):
        """Rename "lm_labels" to "masked_lm_labels" and "lm_weights" to "masked_lm_weights" """
        y_new = {}
        y_new["masked_lm_labels"] = y["lm_labels"]
        y_new["masked_lm_weights"] = y["lm_weights"]
        return x, y_new

    def add_masked_lm_positions(x, y):
        """Add masked lm positions to inputs for prefix and causal LM"""
        x["masked_lm_positions"] = tf.cast(tf.range(tf.shape(x['input_ids'])[0]), tf.int32)
        return x, y

    def remove_type(x, y):
        """Remove type from labels"""
        del y['type']
        return x, y

    def lm_based_on_probability(example):
        """Choode MLM, CLM, PLM based on probability"""

        item = example['text']
        prob = tf.random.uniform(shape=())

        # 30 percent of time, do prefix language modeling
        if prob <= 0.34:
            item_dict = split_text(item)
            item_dict = filter_empty_string(item_dict)
            inputs, labels = prefix_map_fn(item_dict)
            inputs, labels = rename_labels_dict(inputs, labels)
            inputs, labels = add_masked_lm_positions(inputs, labels)

            # Add 3d mask for the model, because its difficult to choose the mask
            # on the fly inside the model
            inputs['input_mask_3d'] = tf.cast(prefix_mask(inputs['input_mask']), tf.float32)

            del inputs['input_mask']
            del inputs['input_type_ids']

            labels['type'] = 'prefix'
            return inputs, labels

        # Causal LM (34-68 percent)
        elif prob < 0.68:
            # Our data has sentences joined by '__||__'. So, for word based MLM
            # we need to replace '__||__', by ''. and club it as a single sentence
            # tf.strings.regex_replace not working as expected
            item = tf.strings.split(item, '__||__')
            item = tf.strings.reduce_join([item], separator=' ')
            # Note about [item], because we need atleast a 1d tensor inside dict
            item_dict = {'text': [item]}
            inputs, labels = causal_lm_map_fn(item_dict)
            inputs, labels = rename_labels_dict(inputs, labels)
            inputs, labels = add_masked_lm_positions(inputs, labels)

            inputs['input_mask_3d'] = attention_mask_square(max_seq_len)

            # del inputs['input_mask']
            del inputs['input_type_ids']

            labels['type'] = 'causal'
            return inputs, labels
        # Remaining do MLM
        else:
            # Our data has sentences joined by '__||__'. So, for word based MLM
            # we need to replace '__||__', by ''. and club it as a single sentence
            # tf.strings.regex_replace not working as expected
            item = tf.strings.split(item, '__||__')
            item = tf.strings.reduce_join([item], separator=' ')
            # Here [item] is not required, as we handle it inside masked_lm_map_fn
            item_dict = {'text': item}
            inputs, labels = masked_lm_map_fn(item_dict)

            inputs['input_mask_3d'] = prepare_3d_input_mask_mlm(tf.expand_dims(inputs['input_mask'], axis=0))

            del inputs['input_mask']
            del inputs['input_type_ids']

            # Cast tf.int64 back to tf.int32, otherwise tf will throw error
            # different types in different branch
            inputs['masked_lm_positions'] = tf.cast(inputs['masked_lm_positions'], tf.int32)
            labels['masked_lm_weights'] = tf.cast(labels['masked_lm_weights'], tf.int32)

            labels['type'] = 'mlm'
            return inputs, labels

    all_text_files = tf.io.gfile.glob(os.path.join(data_directory, '*.txt'))
    shuffle(all_text_files)
    ds = tf.data.TextLineDataset(all_text_files)

    # Remove duplicates if any
    ds = ds.unique()

    # We need to add the text as dict
    ds = ds.map(lambda x: {'text': x}, num_parallel_calls=tf.data.AUTOTUNE)

    # Do MLM
    ds = ds.map(lm_based_on_probability, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter examples if there is not atleast single MASK sentence
    ds = ds.filter(filter_out_empty_mask)

    # Filter examples for prefix where minimum_length doesnt meet
    ds = ds.filter(filter_prefix_with_minimum_length)

    # Remove type from labels, as TPU wont support string
    ds = ds.map(remove_type, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and Prefetch
    ds = ds.shuffle(1024, reshuffle_each_iteration=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Batch to fixed shapes, TPU requires that
    _padded_shapes = (
        {'input_ids': [max_seq_len], 'input_mask_3d': [max_seq_len, max_seq_len], 'masked_lm_positions': [max_seq_len]},
        {
            'masked_lm_labels': [max_seq_len],
            'masked_lm_weights': [max_seq_len],
        },
    )
    ds = ds.padded_batch(batch_size, padded_shapes=_padded_shapes, drop_remainder=True)

    # Auto SHARD
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    ds = ds.with_options(options)

    return ds
