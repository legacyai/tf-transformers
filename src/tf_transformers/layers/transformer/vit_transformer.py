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
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_transformers.core import LegacyLayer
from tf_transformers.layers import dense_einsum
from tf_transformers.layers.attention import MultiHeadAttention
from tf_transformers.utils import tf_utils


class TransformerVIT(LegacyLayer):
    """Transformer

    This layer implements the Transformer from "Attention Is All You Need".
    (https://arxiv.org/abs/1706.03762).

    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        intermediate_activation,
        use_auto_regressive,
        attention_head_size=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_bias=True,
        use_decoder=False,
        share_attention_layers=True,
        layer_norm_epsilon=None,
        is_training=False,
        use_dropout=False,
        name="transformer",
        **kwargs,
    ):
        """
        Args:
            num_attention_heads: int, Number of attention heads.
            intermediate_size: int, Size of the intermediate layer.
            intermediate_activation: keras object, Activation for the intermediate layer.
            attention_cfg: The config with which to instantiate `attention_cls`. Ignored
            if attention_cls is a layer instance.
            dropout_rate: float (between 0 and 1), Dropout probability
                            for the post-attention and output dropout.
            attention_dropout_rate: float (between 0 and 1), Dropout probability
                            for within the attention layer.
            kernel_initializer: Initializer for dense layer kernels.
            bias_initializer: Initializer for dense layer biases.
            kernel_regularizer: Regularizer for dense layer kernels.
            bias_regularizer: Regularizer for dense layer biases.
            activity_regularizer: Regularizer for dense layer activity.
            kernel_constraint: Constraint for dense layer kernels.
            bias_constraint: Constraint for dense layer kernels.
            share_attention_layers: To share same attention layers in decoder cross attentions
            cross_attention_inside_encoder: Whether we want to use cross attention \
                inside encoder.
            is_decoder: bool
        """
        super(TransformerVIT, self).__init__(name=name, is_training=is_training, use_dropout=use_dropout, **kwargs)
        # mostly embedding_size is same as projecting after attention
        self._hidden_size = hidden_size
        self._num_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = intermediate_activation
        self._attention_head_size = attention_head_size
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._use_bias = use_bias
        self._use_decoder = use_decoder
        self._layer_norm_epsilon = layer_norm_epsilon
        self._is_training = is_training
        self._use_dropout = use_dropout
        self._use_auto_regressive = use_auto_regressive

    def build(self, input_shape):
        """Build variables based on shape at run time.

        Args:
            input_shape ([input_word_embeddings 3D, attention_mask 3D]): input_word_embeddings
            (b x s x h) and attention_mask (b x 1 x s)

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        input_tensor = input_shape[0]
        input_tensor_shape = tf.TensorShape(input_tensor)
        batch_size, sequence_length, embedding_size = input_tensor_shape

        if not self._attention_head_size:
            # If attention_head is None, then make sure
            # it can be inferred from (embedding_size // self._num_heads)
            if embedding_size % self._num_heads != 0:
                raise ValueError(
                    "The input size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (embedding_size, self._num_heads)
                )
            self._attention_head_size = int(embedding_size // self._num_heads)

        # Common kwargs
        common_kwargs = dict(
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        # Self Attention Norm
        self._pre_attention_norm = tf.keras.layers.LayerNormalization(
            name="pre_attention_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )

        # Self Attention Layer
        self._attention_layer = MultiHeadAttention(
            num_heads=self._num_heads,
            head_size=self._attention_head_size,
            dropout_rate=self._attention_dropout_rate,
            name="self_attention",
            is_training=self._is_training,
            use_decoder=self._use_decoder,
            use_auto_regressive=self._use_auto_regressive,
            use_dropout=self._use_dropout,
            **common_kwargs,
        )

        # Dense layer
        self._attention_output_dense = dense_einsum.DenseEinsum(
            output_shape=self._hidden_size, name="self_attention_output", **common_kwargs
        )

        # Attention Dropout
        self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

        # Self Attention Norm
        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )

        # Cross Attention for Decoder
        if self._use_decoder:
            # Cross Attention layer
            self._cross_attention_layer = MultiHeadAttention(
                num_heads=self._num_heads,
                head_size=self._attention_head_size,
                dropout_rate=self._attention_dropout_rate,
                name="cross_attention",
                is_training=self._is_training,
                use_auto_regressive=self._use_auto_regressive,
                use_decoder=self._use_decoder,
                use_dropout=self._use_dropout,
                **common_kwargs,
            )
            # Dense
            self._cross_attention_output_dense = dense_einsum.DenseEinsum(
                output_shape=self._hidden_size, name="cross_attention_output", **common_kwargs
            )
            # Norm
            self._cross_attention_layer_norm = tf.keras.layers.LayerNormalization(
                name="cross_attention_layer_norm",
                axis=-1,
                epsilon=self._layer_norm_epsilon,
                dtype=tf.float32,
            )

        # Main Dense Layer after Attention
        self._intermediate_dense = dense_einsum.DenseEinsum(
            output_shape=self._intermediate_size,
            activation=self._intermediate_activation,
            # This layer is always float32 for numeric stability.
            dtype=tf.float32,
            name="intermediate",
            **common_kwargs,
        )
        # intermediate Dense
        self._output_dense = dense_einsum.DenseEinsum(output_shape=self._hidden_size, name="output", **common_kwargs)
        self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

        # Use float32 in layernorm for numeric stability.
        # self._output_layer_norm = tf.keras.layers.LayerNormalization(
        #     name="output_layer_norm", axis=-1, epsilon=self._layer_norm_epsilon, dtype=tf.float32
        # )
        super(TransformerVIT, self).build(input_shape)

    def get_config(self):
        config = {
            "hidden_size": self._hidden_size,
            "num_attention_heads": self._num_heads,
            "intermediate_size": self._intermediate_size,
            "intermediate_activation": self._intermediate_activation,
            "dropout_rate": self._dropout_rate,
            "attention_dropout_rate": self._attention_dropout_rate,
            "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self._bias_constraint),
            "is_training": self.is_training,
            "use_auto_regressive": self._use_auto_regressive,
        }
        base_config = super(TransformerVIT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call_encoder(self, inputs, cache_key=None, cache_value=None):
        """
        Training pipeline
        """

        # b x s x h   # b x s x s
        input_tensor, attention_mask = inputs
        # GPT2 Pre Attention Norm
        input_tensor_norm = self._pre_attention_norm(input_tensor)
        # [from_tensor, to_tensor]
        attention_inputs = [input_tensor_norm, input_tensor_norm]
        if attention_mask is not None:
            attention_inputs.append(attention_mask)

        # attention_inputs = [from_tensor, to_tensor, attention_mask]
        attention_output, key, value = self._attention_layer(
            attention_inputs, cache_key=cache_key, cache_value=cache_value
        )
        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self._use_dropout)
        attention_output_copy = tf.identity(attention_output)
        attention_output = self._attention_layer_norm(input_tensor + attention_output)
        # mixed precision stability requires Normalization to be in tf.ffloat32
        attention_output = tf.cast(attention_output, dtype=tf_utils.get_dtype())
        intermediate_output = self._intermediate_dense(attention_output)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        return layer_output + input_tensor + attention_output_copy, key, value

    def call_decoder(self, inputs, cache_key=None, cache_value=None):
        """
        Training pipeline
        """
        input_tensor, attention_mask, encoder_output, decoder_encoder_mask = inputs

        # Decoder Self Attention
        input_tensor_norm = self._pre_attention_norm(input_tensor)
        attention_inputs = [input_tensor_norm, input_tensor_norm]
        if attention_mask is not None:
            attention_inputs.append(attention_mask)
        attention_output, key, value = self._attention_layer(
            attention_inputs, cache_key=cache_key, cache_value=cache_value
        )

        # Self Attention Dense + Norm
        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
        attention_output = self._attention_layer_norm(attention_output + input_tensor)

        if self._use_decoder:
            # Cross Attention
            attention_output_copy = tf.identity(attention_output, name="attention_output_copy")
            # No pre norm for decoder to encoder attention
            attention_inputs_for_decoder = [
                attention_output_copy,
                encoder_output,
                decoder_encoder_mask,
            ]
            # For auto-regressive we need this
            # cache_key has to be zeros, because nothng
            # to cache in cross_attention
            cache_key_cross = None
            cache_value_cross = None
            if cache_key is not None and self._use_auto_regressive:
                cache_key_cross = tf.zeros_like(cache_key)
                cache_value_cross = tf.zeros_like(cache_value)
            attention_output, _, _ = self._cross_attention_layer(
                attention_inputs_for_decoder, cache_key=cache_key_cross, cache_value=cache_value_cross
            )

            attention_output = self._cross_attention_output_dense(attention_output)
            attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
            attention_output_copy = tf.cast(attention_output_copy, dtype=tf_utils.get_dtype())
            attention_output = self._cross_attention_layer_norm(attention_output_copy + attention_output)
            attention_output = tf.cast(attention_output, dtype=tf_utils.get_dtype())
        # Last Projection
        intermediate_output = self._intermediate_dense(attention_output)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        layer_output = tf.cast(layer_output, dtype=tf_utils.get_dtype())
        return layer_output, key, value

    def call(self, inputs, mode="encoder", cache_key=None, cache_value=None):
        """Call

        Args:
            inputs ([embeddings 3D, attention_mask 3D]): List of [embeddings,
                                                                attention_mask]
            mode (str, optional): [description]. Defaults to "encoder".
            cache_key ([type], optional): [description]. Defaults to None.
            cache_value ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if self._use_decoder:
            outputs = self.call_decoder(inputs, cache_key=cache_key, cache_value=cache_value)
        else:
            outputs = self.call_encoder(inputs, cache_key=cache_key, cache_value=cache_value)
        return outputs
