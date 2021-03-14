# Copyright 2019 The TensorFlow/tf_transformers Authors. All Rights Reserved.
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
"""Keras-based GPT2 Layer Normalization layer."""
# from __future__ import google_type_annotations
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_transformers.utils import tf_utils


@tf.keras.utils.register_keras_serializable(package="Text")
class GPT2LayerNormalization(tf.keras.layers.Layer):
    """Creates a GPT2 Layer Normalization.

    This layer creates a LayerNormalization as described in
    https://github.com/openai/gpt-2/blob/master/src/model.py#L28

    This layer can be used to Normalize to mean = 0, std = 1, then do a diagonal affine transform.

    Arguments:
    """

    def __init__(
        self,
        initializer="constant",
        beta_initializer="zeros",
        gamma_initializer="ones",
        axis=-1,
        epsilon=1e-5,
        **kwargs,
    ):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super(GPT2LayerNormalization, self).__init__(**kwargs)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        """Implements build() for the layer."""
        dimension_list = input_shape.as_list()

        if len(dimension_list) != 3:
            raise ValueError(
                "GPT2LayerNormalization expects a 3-dimensional input tensor " "of shape [batch, sequence, width]"
            )
        # seq_length = dimension_list[1]
        width = dimension_list[2]

        self.gamma = self.add_weight("gamma", shape=[width], initializer=self.gamma_initializer)
        self.beta = self.add_weight("beta", shape=[width], initializer=self.beta_initializer)

        super(GPT2LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        """Implements call() for the layer."""

        u = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(tf.square(inputs - u), axis=self.axis, keepdims=True)
        inputs = (inputs - u) * tf.math.rsqrt(s + self.epsilon)
        inputs = inputs * self.gamma + self.beta
        return inputs

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
        }
        base_config = super(GPT2LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="Text")
class T5LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, initializer="constant", beta_initializer="ones", epsilon=1e-6, axis=-1, **kwargs):
        """Construct a layernorm module in the T5 style
        No bias and no substraction of mean.
        """
        super(T5LayerNormalization, self).__init__(**kwargs)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.variance_epsilon = epsilon
        self.axis = axis

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer=self.beta_initializer)
        super(T5LayerNormalization, self).build(input_shape)

    def call(self, x):
        variance = tf.reduce_mean(tf.square(x), axis=self.axis, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * x

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.variance_epsilon,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
        }
        base_config = super(T5LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
