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
import tensorflow as tf
from absl import logging

from tf_transformers.data.utils import separate_x_y

logging.set_verbosity("INFO")


class TFProcessor(object):
    """
    TFProcessor class . This class is responsible to read data, \
    and convert it to a tf.data.Dataset

    """

    def auto_batch(
        self,
        tf_dataset,
        batch_size,
        padded_values=None,
        padded_shapes=None,
        x_keys=None,
        y_keys=None,
        shuffle=False,
        drop_remainder=False,
        shuffle_buffer_size=100,
        prefetch_buffer_size=100,
    ):
        """Auto Batching

        Args:
            tf_dataset (tf.data.Dataset): TF dataset
            batch_size (int): Batch Size
            padded_values (dict): dict of key to padded values eg: {'key': tf.constant(0)}
            padded_shapes (dict): dict of key to padded shapes eg: 'key': (None,)}
            x_keys (list): List of key names. We will filter based on this.
            y_keys (list): List of key names.
            shuffle (bool):  Defaults to False.
            shuffle_buffer_size (int):  Defaults to 100.
            prefetch_buffer_size (int): Defaults to 100.

        Returns:
            tf.data.Dataset: Batched
        """
        element_spec = tf_dataset.element_spec
        _padded_values = {}
        if not padded_values:
            padded_values = {}
        if not padded_shapes:
            padded_shapes = {}
        # sometimes we might have to have sme custom values other than 0
        for k, v in element_spec.items():
            if k in padded_values:
                value = padded_values[k]
                _padded_values[k] = tf.constant(value, dtype=value.dtype)
            else:
                if v.dtype == tf.string:
                    _padded_values[k] = tf.constant("0", dtype=v.dtype)
                    continue

                _padded_values[k] = tf.constant(0, dtype=v.dtype)

        _padded_shapes = {}
        for k, v in element_spec.items():
            if k in padded_shapes:
                _padded_shapes[k] = padded_shapes[k]
            else:
                if len(v.shape.dims) == 1:
                    _padded_shapes[k] = [None]
                if len(v.shape.dims) == 0:
                     _padded_shapes[k] = []
                if len(v.shape.dims) > 1:
                    raise ValueError("Seems like `{}` has 2 dimensional or more".format(v))

        dataset = tf_dataset.padded_batch(
            padding_values=_padded_values,
            padded_shapes=_padded_shapes,
            batch_size=batch_size,
            drop_remainder=drop_remainder,
        )
        # fmt: off
        if x_keys and y_keys:
            dataset = dataset.map(lambda x: separate_x_y(x, x_keys, y_keys), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # noqa
        # fmt: on
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def process(self, parse_fn, verbose=10000):
        """This function will iterate over parse_fn and keep writing it TFRecord
        Args:

            parse_fn: function which should be an iterator or generator
        """
        data = {}
        if hasattr(parse_fn, "__iter__") and not hasattr(parse_fn, "__len__"):
            counter = 0
            for entry in parse_fn:
                for k, v in entry.items():
                    if k in data:
                        data[k].append(v)
                    else:
                        data[k] = [v]
                counter += 1

                if counter % verbose == 0:
                    logging.info("Processed  {} examples so far".format(counter))

            logging.info("Total individual observations/examples written is {}".format(counter))
            data_ragged = {k: tf.ragged.constant(v) for k, v in data.items()}
            dataset = tf.data.Dataset.from_tensor_slices(data_ragged)
            return dataset
        else:
            raise ValueError("Expected `parse_fn` to be a generator/iterator ")
