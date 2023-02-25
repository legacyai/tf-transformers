# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common TF utilities."""

from __future__ import absolute_import, division, print_function

import six
import tensorflow as tf
from tensorflow.python.util import deprecation

from tf_transformers import activations


@deprecation.deprecated(
    None,
    "tf.keras.layers.Layer supports multiple positional args and kwargs as "
    "input tensors. pack/unpack inputs to override __call__ is no longer "
    "needed.",
)
def pack_inputs(inputs):
    """Pack a list of `inputs` tensors to a tuple.

    Args:
      inputs: a list of tensors.

    Returns:
      a tuple of tensors. if any input is None, replace it with a special constant
      tensor.
    """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if x is None:
            outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
        else:
            outputs.append(x)
    return tuple(outputs)


@deprecation.deprecated(
    None,
    "tf.keras.layers.Layer supports multiple positional args and kwargs as "
    "input tensors. pack/unpack inputs to override __call__ is no longer "
    "needed.",
)
def unpack_inputs(inputs):
    """unpack a tuple of `inputs` tensors to a tuple.

    Args:
      inputs: a list of tensors.

    Returns:
      a tuple of tensors. if any input is a special constant tensor, replace it
      with None.
    """
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if is_special_none_tensor(x):
            outputs.append(None)
        else:
            outputs.append(x)
    x = tuple(outputs)

    # To trick the very pointless 'unbalanced-tuple-unpacking' pylint check
    # from triggering.
    if len(x) == 1:
        return x[0]
    return tuple(outputs)


def is_special_none_tensor(tensor):
    """Checks if a tensor is a special None Tensor."""
    return tensor.shape.ndims == 0 and tensor.dtype == tf.int32


# TODO(hongkuny): consider moving custom string-map lookup to keras api.
def get_activation(identifier):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

    It checks string first and if it is one of customized activation not in TF,
    the corresponding activation will be returned. For non-customized activation
    names and callable identifiers, always fallback to tf.keras.activations.get.

    Args:
      identifier: String name of the activation function or callable.

    Returns:
      A Python function corresponding to the activation function.
    """
    if isinstance(identifier, six.string_types):
        name_to_fn = {
            "gelu": activations.gelu,
            "simple_swish": activations.simple_swish,
            "hard_swish": activations.hard_swish,
            "identity": activations.identity,
        }
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
            "equal to the expected tensor rank `%s`" % (name, actual_rank, str(tensor.shape), str(expected_rank))
        )


def get_dtype():
    dtype = tf.float32
    try:
        policy = tf.keras.mixed_precision.experimental.global_policy()
    except:
        policy = tf.keras.mixed_precision.global_policy()
    if policy.name == "mixed_float16":
        dtype = tf.float16
    if policy.name == "mixed_bfloat16":
        dtype = tf.bfloat16
    return dtype


def gather_values_from_2d_tensor(value_tensor, index_tensor):
    """Get values from 2D tensor using 2D index

    value_tensor:
        tf.Tensor: shape=(2, 10), dtype=float32, numpy=
        array([[0.41716623, 0.16220212, 0.57236147, 0.85827255, 0.07229817,
                0.7548058 , 0.34538198, 0.50186884, 0.17406607, 0.6326196 ],
                [0.50965476, 0.7139858 , 0.66155374, 0.77050793, 0.56380427,
                0.80631006, 0.81072354, 0.17372155, 0.742455  , 0.46470654]],
        dtype=float32)>
    index_tensor:
        <tf.Tensor: shape=(2, 1), dtype=int64, numpy=
        array([[1],
             [5]])>

    Values based on index in each row has to be returned.
    Returns (value_tensor[0][1], value_tensor[1][5])

    Args:
        value_tensor (tf.Tensor): 2D Tensor of (Matrxi) values
        index_tensor (tf.Tensor): 2D tensor of indexes ( batch_size x 1)

    Returns:
        tf.Tensor: 1D (batch_size,)
    """
    batch_size = tf.shape(index_tensor)[0]  # scalar
    batch_range = tf.expand_dims(tf.range(batch_size), 1)  # 2d (batch_size, 1)
    index_tensor_2d = tf.concat(
        [batch_range, tf.cast(index_tensor, dtype=batch_range.dtype)], axis=1
    )  # 2D (batch_size, 2)
    return tf.gather_nd(value_tensor, index_tensor_2d)  # (batch_size, )


def is_gpu_available():
    counter = 0
    devices = tf.config.list_physical_devices()
    for device in devices:
        if 'GPU' in device.name:
            counter += 1
    if counter == 0:
        return False, counter
    return True, counter
