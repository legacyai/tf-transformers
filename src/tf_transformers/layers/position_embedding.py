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

"""Keras-based positional embedding layer."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="legacyai.text")
class PositionEmbedding(tf.keras.layers.Layer):
    """Creates a positional embedding.

    This layer creates a positional embedding as described in "BERT":
    (https://arxiv.org/abs/1810.04805).
    """

    def __init__(
        self,
        max_sequence_length,
        embedding_width,
        initializer="glorot_uniform",
        name="positional_embeddings",
        dtype=tf.float32,
        **kwargs,
    ):
        """
        Args:
            max_sequence_length ([int]): Max allowed sequence length (512 in BERT)
            embedding_width ([type]): Output size of the embedding layer.
            initializer (str, optional): The initializer to use for the
            embedding weights. Defaults to "glorot_uniform".
            name (str, optional): name of the layer. Defaults to "positional_embeddings".
            dtype ([type], optional): [description]. Defaults to tf.float32.
        """
        super(PositionEmbedding, self).__init__(name=name, dtype=dtype, **kwargs)
        self._max_sequence_length = max_sequence_length
        self._embedding_width = embedding_width
        self._initializer = initializer

    def get_config(self):
        """Config based on init arguments

        Returns:
            [dict]: Dict of all init arguments
        """
        config = {
            "max_sequence_length": self._max_sequence_length,
            "embedding_width": self._embedding_width,
            "initializer": self._initializer,
            "name": self._name,
            "dtype": self._dtype,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Build embeddings on run time (once)

        Args:
            input_shape ([TensorShape or List of TensorShape]): Shape of inputs
        """
        self.embeddings = self.add_weight(
            "embeddings",
            shape=[self._max_sequence_length, self._embedding_width],
            initializer=self._initializer,
        )

        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        """Call

        Args:
            inputs ([tf.Tensor]): input ids 1D (tf.range(sequence_length))

        Returns:
            [tf.Tensor]: embeddings 3D (b x s x h)
        """
        # REPLACED (tf.identity for TFlite)
        position_embeddings = tf.gather(tf.identity(self.embeddings), inputs)
        position_embeddings = tf.expand_dims(position_embeddings, 0)
        return position_embeddings
