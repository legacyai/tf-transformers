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


def attention_mask_square(nd, *, dtype=tf.float32):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    ns = nd
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


@tf.keras.utils.register_keras_serializable(package="Text")
class CausalMask(tf.keras.layers.Layer):
    """Create 3D attention mask from a 3D tensor mask.

    inputs[0]: from_tensor: 2D or 3D Tensor of shape
      [batch_size, from_seq_length, ...].
    inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """

    def call(self, inputs):
        from_tensor = inputs
        from_shape = tf_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        # 2D Lower Triangular Mask
        from_mask = attention_mask_square(from_seq_length)

        # Replicate 2D `N` times
        mask = tf.ones([batch_size, 1, 1]) * from_mask

        return mask
