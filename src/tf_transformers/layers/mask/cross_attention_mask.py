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
"""Keras layer that creates a self-attention mask."""

# from __future__ import google_type_annotations
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_transformers.utils import tf_utils


@tf.keras.utils.register_keras_serializable(package="Text")
class CrossAttentionMask(tf.keras.layers.Layer):
    """Create 3D attention mask from a 2D tensor mask.

    inputs[0]: from_tensor: 2D or 3D Tensor of shape
      [batch_size, from_seq_length, ...].
    inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """

    def __init__(self, **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super(CrossAttentionMask, self).__init__(**kwargs)
        self._dtype = kwargs["dtype"]

    def call(self, inputs):
        to_mask = inputs[1]
        batch_size, from_seq_length = tf_utils.get_shape_list(inputs[0])
        _, to_seq_length = tf_utils.get_shape_list(inputs[1])

        to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=self._dtype)

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=self._dtype)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask
