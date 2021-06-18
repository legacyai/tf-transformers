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

"""Keras-based one-hot embedding layer."""

import tensorflow as tf

from tf_transformers.utils import tf_utils


@tf.keras.utils.register_keras_serializable(package="legacyai.text")
class OnDeviceEmbedding(tf.keras.layers.Layer):
    """Performs an embedding lookup suitable for accelerator devices.

    This layer uses either tf.gather or tf.one_hot to translate integer indices to
    float embeddings.
    """

    def __init__(
        self,
        vocab_size,
        embedding_width,
        initializer="glorot_uniform",
        use_one_hot=True,
        name="word_embedding",
        dtype=tf.float32,
        trainable=True,
        **kwargs,
    ):
        """
        Args:

            vocab_size ([int]): Number of elements in the vocabulary
            embedding_width ([int]): Output size of the embedding layer.
            initializer (str, optional): The initializer to use for the
            embedding weights. Defaults to "glorot_uniform".
            use_one_hot (bool,optional): Whether to use tf.one_hot over tf.gather for the
            embedding. Defaults to False (that is, using tf.gather).
            name (str,optional): Name of the layer. Defaults to "word_embedding".
            dtype([type], optional): Defaults to tf.float32.
            trainable (bool,optional): Defaults to True.
        """

        super(OnDeviceEmbedding, self).__init__(name=name, dtype=dtype, trainable=trainable, **kwargs)
        self._vocab_size = vocab_size
        self._embedding_width = embedding_width
        self._initializer = initializer
        self._use_one_hot = use_one_hot
        self._name = name
        self._dtype = dtype
        self._trainable = trainable

    def get_config(self):
        """Config based on init arguments

        Returns:
            [dict]: Dict of all init arguments
        """
        config = {
            "vocab_size": self._vocab_size,
            "embedding_width": self._embedding_width,
            "initializer": self._initializer,
            "use_one_hot": self._use_one_hot,
            "name": self._name,
            "dtype": self._dtype,
            "trainable": self._trainable,
        }
        base_config = super(OnDeviceEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Build embeddings on run time (once)

        Args:
            input_shape ([TensorShape or List of TensorShape]): Shape of inputs
        """
        self.embeddings = self.add_weight(
            "embeddings",
            shape=[self._vocab_size, self._embedding_width],
            initializer=self._initializer,
            dtype=self._dtype,
        )

        super(OnDeviceEmbedding, self).build(input_shape)

    def call(self, inputs):
        """Call

        Args:
            inputs ([tf.Tensor]): input ids (b x s)

        Returns:
            [tf.Tensor]: embeddings (b x s x h)
        """
        input_shape = tf_utils.get_shape_list(inputs, expected_rank=2)
        input_shape.append(self._embedding_width)  # b x s x h
        flat_inputs = tf.reshape(inputs, [-1])
        if self._use_one_hot:
            one_hot_data = tf.one_hot(flat_inputs, depth=self._vocab_size, dtype=self._dtype)
            embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            # REPLACED (tf.identity for TFlite)
            # embeddings = tf.gather(self.embeddings, flat_inputs)
            embeddings = tf.identity(tf.gather(self.embeddings, flat_inputs))
        embeddings = tf.reshape(embeddings, input_shape)

        return embeddings
