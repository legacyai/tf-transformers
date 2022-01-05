# coding=utf-8
# Copyright 2021 TF-Transformers Authors.
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
"""A simple MLM layer  in Tensorflow 2.0"""
import tensorflow as tf

from tf_transformers.activations import get_activation


@tf.keras.utils.register_keras_serializable(package="legacyai.text")
class MaskedLM(tf.keras.layers.Layer):
    """Masked language model network head for BERT modeling.
    This layer implements a masked language model based on the provided
    transformer based encoder.
    """

    def __init__(
        self, hidden_size, layer_norm_epsilon, activation=None, initializer="glorot_uniform", name=None, **kwargs
    ):
        """
        Args:
            hidden_size ([int]): Embedding size
            layer_norm_epsilon ([float]): layer Norm epsilon value
            activation ([str], optional): Activation
            initializer (str, optional): [description]. Defaults to 'glorot_uniform'.
            name ([type], optional): [description]. Defaults to None.
        """
        super(MaskedLM, self).__init__(name=name, **kwargs)
        self._hidden_size = hidden_size
        self._layer_norm_epsilon = layer_norm_epsilon
        self._activation = activation
        self._initializer = initializer
        if activation == "gelu":
            self.activation = get_activation(activation)
        else:
            self.activation = activation

    def build(self, input_shape):
        """Build variables dynamically.(one time)

        Args:
            input_shape ([tf.Tensor]):
        """
        self.dense = tf.keras.layers.Dense(
            self._hidden_size, activation=self.activation, kernel_initializer=self._initializer, name="transform/dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=self._layer_norm_epsilon, name="transform/LayerNorm"
        )

        super(MaskedLM, self).build(input_shape)

    def call(self, sequence_data, masked_positions=None):
        """Call

        Args:
            sequence_data ([tf.Tensor (tf.float32)]): Token Embeddings # b x s x h
            masked_positions ([tf.Tensor (tf.int32)], optional):
            When doing MaskedLM, we only have to find logits of masked tokens.
            Because, even if we calculate full logits, eventually we mask it off.
            Defaults to None.

        Returns:
            [tf.Tensor]: token logits # b x s x vocab_size
        """
        if masked_positions is not None:
            sequence_data = self._gather_indexes(sequence_data, masked_positions)
        lm_data = self.dense(sequence_data)
        lm_data = self.layer_norm(lm_data)
        return lm_data
        lm_data = tf.matmul(lm_data, self.embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(lm_data, self.bias)
        return logits

    def get_config(self):
        """Config based on init arguments

        Returns:
            [dict]: Dict of all init arguments
        """
        config = {
            "hidden_size": self._hidden_size,
            "layer_norm_epsilon": self._layer_norm_epsilon,
            "initializer": self._initializer,
            "activation": self._activation,
            "name": self._name,
        }
        base_config = super(MaskedLM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions, for performance.
        Args:
            sequence_tensor: Sequence output of shape
                (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
                hidden units.
            positions: Positions ids of tokens in sequence to mask for pretraining
                of with dimension (batch_size, num_predictions) where
                `num_predictions` is maximum number of tokens to mask out and predict
                per each sequence.
        Returns:
            Masked out sequence tensor of shape (batch_size * num_predictions,
            num_hidden).
        """
        sequence_shape = tf.shape(sequence_tensor)
        batch_size, seq_length = sequence_shape[0], sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        # Make it 3D (b x s x h)
        output_tensor = tf.reshape(output_tensor, (batch_size, -1, sequence_tensor.shape[2]))
        return output_tensor
