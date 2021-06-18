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
from functools import wraps

import tensorflow as tf


def auto_batch(
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
            _padded_shapes[k] = [None]
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


def pad_dataset(tokenizer_fn):
    """We will pad the data.
    Based on name of the dataset, we will pad it accordingly

    Args:
        tokenizer_fn ([type]): [A function which returns dict of list of list]

    Returns:
        [type]: [description]
    """

    @wraps(tokenizer_fn)
    def pad_fn(*args, **kw):
        tokenized_dict = tokenizer_fn(*args, **kw)
        tokenized_dict_ragged = {name: tf.ragged.constant(tensor) for name, tensor in tokenized_dict.items()}
        tokenized_dict_padded = {}
        for name, tensor in tokenized_dict_ragged.items():
            if isinstance(tensor, tf.RaggedTensor):
                if name in ["input_ids", "encoder_input_ids"]:
                    tokenized_dict_padded[name] = tensor.to_tensor(-1)
                elif name in [
                    "input_mask",
                    "input_type_ids",
                    "encoder_input_mask",
                    "encoder_input_type_ids",
                ]:
                    tokenized_dict_padded[name] = tensor.to_tensor(0)
            else:
                tokenized_dict_padded[name] = tensor
        return tokenized_dict_padded

    return pad_fn


def pad_dataset_normal(tokenizer_fn):
    """We will pad the data.
    Based on name of the dataset, we will pad it accordingly

    Args:
        tokenizer_fn ([type]): [A function which returns dict of list of list]

    Returns:
        [type]: [description]
    """

    @wraps(tokenizer_fn)
    def pad_fn(*args, **kw):
        tokenized_dict = tokenizer_fn(*args, **kw)
        tokenized_dict_ragged = {name: tf.ragged.constant(tensor) for name, tensor in tokenized_dict.items()}
        tokenized_dict_padded = {}
        for name, tensor in tokenized_dict_ragged.items():
            if isinstance(tensor, tf.RaggedTensor):
                if name in ["input_ids", "encoder_input_ids"]:
                    tokenized_dict_padded[name] = tensor.to_tensor(0)
                elif name in [
                    "input_mask",
                    "input_type_ids",
                    "encoder_input_mask",
                    "encoder_input_type_ids",
                ]:
                    tokenized_dict_padded[name] = tensor.to_tensor(0)
            else:
                tokenized_dict_padded[name] = tensor
        return tokenized_dict_padded

    return pad_fn


def separate_x_y(dict, x_keys, y_keys):
    """Separate dataset into a tuple (X, Y)

    Args:
        dict ([type]): Each entry in tf dataset
        x_keys ([type]): List of key values
        y_keys ([type]): List of ky values

    Returns:
        tuple of each entry
    """
    X = {}
    Y = {}
    for k, v in dict.items():
        if k in x_keys:
            X[k] = v
            continue
        if k in y_keys:
            Y[k] = v
    return (X, Y)


def pad_ragged(dataset):
    """
    Pad dataset of dict .

    """
    dataset_padded = {}
    for item, tensor in dataset.items():
        if isinstance(tensor, tf.RaggedTensor):
            dataset_padded[item] = tensor.to_tensor()
        else:
            dataset_padded[item] = tensor
    return dataset_padded
