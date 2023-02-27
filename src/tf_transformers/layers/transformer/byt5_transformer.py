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
from tf_transformers.layers import T5LayerNormalization, dense_einsum
from tf_transformers.layers.attention import T5Attention
from tf_transformers.utils import tf_utils


class TransformerByT5(LegacyLayer):
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
        bidirectional,
        create_positonal_embedding,
        positional_buckets,
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
        use_bias=False,
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
            bidirectional: bool , For relative positional embeddings
            create_positonal_embedding: bool, T5 uses it only at 1st layer
            attention_head_size: size of attention head
            positional_buckets: int, For relative positional embedding
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
        super(TransformerByT5, self).__init__(name=name, is_training=is_training, use_dropout=use_dropout, **kwargs)
        # mostly embedding_size is same as projecting after attention
        self._hidden_size = hidden_size
        self._num_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = intermediate_activation
        self._bidirectional = bidirectional
        self._create_positonal_embedding = create_positonal_embedding
        self._positional_buckets = positional_buckets
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
            use_bias=self._use_bias,
        )

        self._pre_attention_norm = T5LayerNormalization(
            name="pre_attention_norm", axis=-1, epsilon=self._layer_norm_epsilon, dtype=tf.float32
        )

        # Self Attention Layer
        self._attention_layer = T5Attention(
            num_heads=self._num_heads,
            head_size=self._attention_head_size,
            bidirectional=self._bidirectional,
            create_positonal_embedding=self._create_positonal_embedding,
            positional_buckets=self._positional_buckets,
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
        self._attention_layer_norm = T5LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )

        # Cross Attention for Decoder
        if self._use_decoder:
            self._pre_cross_attention_norm = T5LayerNormalization(
                name="pre_cross_attention_norm",
                axis=-1,
                epsilon=self._layer_norm_epsilon,
                dtype=tf.float32,
            )
            # Cross Attention layer
            self._cross_attention_layer = T5Attention(
                num_heads=self._num_heads,
                head_size=self._attention_head_size,
                bidirectional=self._bidirectional,
                create_positonal_embedding=self._create_positonal_embedding,
                positional_buckets=self._positional_buckets,
                dropout_rate=self._attention_dropout_rate,
                name="cross_attention",
                is_training=self._is_training,
                use_decoder=self._use_decoder,
                use_auto_regressive=self._use_auto_regressive,
                use_dropout=self._use_dropout,
                **common_kwargs,
            )
            # Dense
            self._cross_attention_output_dense = dense_einsum.DenseEinsum(
                output_shape=self._hidden_size, name="cross_attention_output", **common_kwargs
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

        self._intermediate_dense2 = dense_einsum.DenseEinsum(
            output_shape=self._intermediate_size,
            activation=None,
            # This layer is always float32 for numeric stability.
            dtype=tf.float32,
            name="intermediate2",
            **common_kwargs,
        )

        # intermediate Dense
        self._output_dense = dense_einsum.DenseEinsum(output_shape=self._hidden_size, name="output", **common_kwargs)
        self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
        super(TransformerByT5, self).build(input_shape)

    def get_config(self):
        config = {
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
        }
        base_config = super(TransformerByT5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call_encoder(self, inputs, position_bias=None, cache_key=None, cache_value=None):
        """
        Encoder pipeline
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
        attention_output, position_bias, key, value = self._attention_layer(
            attention_inputs,
            position_bias=position_bias,
            cache_key=cache_key,
            cache_value=cache_value,
        )
        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self._use_dropout)

        attention_output_copy = tf.identity(attention_output)
        attention_output = self._attention_layer_norm(input_tensor + attention_output)
        # mixed precision stability requires Normalization to be in tf.ffloat32
        attention_output = tf.cast(attention_output, dtype=tf_utils.get_dtype())
        intermediate_output_gelu  = self._intermediate_dense(attention_output)

        intermediate_output = self._intermediate_dense2(attention_output)

        intermediate_output = intermediate_output_gelu * intermediate_output
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)

        return layer_output + input_tensor + attention_output_copy, position_bias, key, value

    def call_decoder(
        self, inputs, position_bias=None, decoder_encoder_position_bias=None, cache_key=None, cache_value=None
    ):
        """
        Decoder pipeline
        """
        input_tensor, attention_mask, encoder_output, decoder_encoder_mask = inputs

        # Decoder Self Attention
        input_tensor_norm = self._pre_attention_norm(input_tensor)
        attention_inputs = [input_tensor_norm, input_tensor_norm]
        if attention_mask is not None:
            attention_inputs.append(attention_mask)
        attention_output, position_bias, key, value = self._attention_layer(
            attention_inputs,
            position_bias=position_bias,
            cache_key=cache_key,
            cache_value=cache_value,
            cross_decoder_mode=False,
        )
        # Self Attention Dense + Norm
        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
        attention_output = input_tensor + attention_output
        attention_output = tf.cast(attention_output, dtype=tf_utils.get_dtype())
        attention_output_copy = tf.identity(attention_output)

        attention_output_norm = self._pre_cross_attention_norm(attention_output)
        attention_inputs_for_decoder = [
            attention_output_norm,
            encoder_output,
            decoder_encoder_mask,
        ]
        (attention_output, decoder_encoder_position_bias, _, _,) = self._cross_attention_layer(
            attention_inputs_for_decoder,
            position_bias=decoder_encoder_position_bias,
            cache_key=cache_key,
            cache_value=cache_value,
            cross_decoder_mode=True,
        )

        attention_output = self._cross_attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
        attention_output = tf.cast(attention_output, dtype=tf_utils.get_dtype())
        attention_output = attention_output + attention_output_copy

        attention_output_normed = self._attention_layer_norm(attention_output)
        intermediate_output_gelu  = self._intermediate_dense(attention_output_normed)
        intermediate_output = self._intermediate_dense2(attention_output_normed)
        intermediate_output = intermediate_output_gelu * intermediate_output
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        layer_output = tf.cast(layer_output, dtype=tf_utils.get_dtype())

        return (
            layer_output + attention_output,
            position_bias,
            decoder_encoder_position_bias,
            key,
            value,
        )

    def call(
        self,
        inputs,
        encoder_output=None,
        decoder_encoder_mask=None,
        position_bias=None,
        decoder_encoder_position_bias=None,
        cache_key=None,
        cache_value=None,
    ):

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
            outputs = self.call_decoder(
                inputs,
                position_bias=position_bias,
                decoder_encoder_position_bias=decoder_encoder_position_bias,
                cache_key=cache_key,
                cache_value=cache_value,
            )
        else:
            outputs = self.call_encoder(
                inputs, position_bias=position_bias, cache_key=cache_key, cache_value=cache_value
            )
        return outputs
