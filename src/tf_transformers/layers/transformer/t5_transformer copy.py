# pylint: disable=g-classes-have-attributes
# from __future__ import google_type_annotations
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_transformers.core import LegacyLayer
from tf_transformers.layers import T5LayerNormalization, dense_einsum
from tf_transformers.layers.attention import T5Attention


class TransformerT5(LegacyLayer):
    """Transformer T5

    This layer implements the Transformer from "Attention Is All You Need".
    (https://arxiv.org/abs/1706.03762), with a customizable attention layer
    option. Users can pass a class to `attention_cls` and associated config to
    `attention_cfg`, in which case the scaffold will instantiate the class with
    the config, or pass a class instance to `attention_cls`.

    """

    def __init__(
        self,
        num_attention_heads,
        intermediate_size,
        intermediate_activation,
        bidirectional,
        create_positonal_embedding,
        positional_buckets,
        attention_head_size=None,
        attention_cfg=None,
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
        is_decoder=False,
        share_attention_layers=False,
        layer_norm_epsilon=None,
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
            pipeline_mode: [auto-regressive, None].
            share_attention_layers: To share same attention layers in decoder cross attentions
            is_decoder: bool
        """
        kwargs["name"] = name
        super(TransformerT5, self).__init__(**kwargs)
        self._attention_cfg = attention_cfg
        self._num_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = intermediate_activation
        self._bidirectional = bidirectional
        self._create_positonal_embedding = create_positonal_embedding
        self._positional_buckets = positional_buckets
        self._attention_dropout_rate = attention_dropout_rate
        self._dropout_rate = dropout_rate
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._use_bias = use_bias
        self._is_decoder = is_decoder
        self._layer_norm_epsilon = layer_norm_epsilon
        self._share_attention_layers = share_attention_layers
        self._attention_head_size = attention_head_size

    def build(self, input_shape):
        """
        Args:
            input_shape: [word_embeddings (3D), attention_mask(3D)]
        """
        input_tensor = input_shape[0]
        input_tensor_shape = tf.TensorShape(input_tensor)
        if len(input_tensor_shape) != 3:
            raise ValueError("TransformerT5 expects a three-dimensional input of " "shape [batch, sequence, width].")
        batch_size, sequence_length, hidden_size = input_tensor_shape

        if not self._attention_head_size:
            if hidden_size % self._num_heads != 0:
                raise ValueError(
                    "The input size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, self._num_heads)
                )
            self._attention_head_size = int(hidden_size // self._num_heads)

        self._pre_attention_norm = T5LayerNormalization(
            name="pre_attention_norm", axis=-1, epsilon=self._layer_norm_epsilon, dtype=tf.float32
        )

        self._attention_layer = T5Attention(
            num_heads=self._num_heads,
            head_size=self._attention_head_size,
            bidirectional=self._bidirectional,
            create_positonal_embedding=self._create_positonal_embedding,
            positional_buckets=self._positional_buckets,
            dropout_rate=self._attention_dropout_rate,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            is_training=self.is_training,
            use_dropout=self.use_dropout,
            name="self_attention",
            use_bias=self._use_bias,
        )

        self._attention_output_dense = dense_einsum.DenseEinsum(
            output_shape=hidden_size,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="self_attention_output",
            use_bias=self._use_bias,
        )

        if self._is_decoder:

            if self._share_attention_layers:
                self._pre_cross_attention_norm = self._pre_attention_norm
                self._cross_attention_layer = self._attention_layer
                self._cross_attention_output_dense = self._attention_output_dense
            else:
                self._pre_cross_attention_norm = T5LayerNormalization(
                    name="pre_cross_attention_norm",
                    axis=-1,
                    epsilon=self._layer_norm_epsilon,
                    dtype=tf.float32,
                )
                # Cross Attention Layer should always work under one mode `is_training = True`.
                # Because encoder_output is fixed.
                # Nothing to concat tenchincally like (K, V) in GPT2
                self._cross_attention_layer = T5Attention(
                    num_heads=self._num_heads,
                    head_size=self._attention_head_size,
                    bidirectional=self._bidirectional,
                    create_positonal_embedding=self._create_positonal_embedding,
                    positional_buckets=self._positional_buckets,
                    dropout_rate=self._attention_dropout_rate,
                    kernel_initializer=self._kernel_initializer,
                    bias_initializer=self._bias_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer,
                    activity_regularizer=self._activity_regularizer,
                    kernel_constraint=self._kernel_constraint,
                    bias_constraint=self._bias_constraint,
                    is_training=self.is_training,
                    use_dropout=self.use_dropout,
                    name="cross_attention",
                    use_bias=self._use_bias,
                    is_cross_attention=True,  # hard code
                )

                self._cross_attention_output_dense = dense_einsum.DenseEinsum(
                    output_shape=hidden_size,
                    kernel_initializer=self._kernel_initializer,
                    bias_initializer=self._bias_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer,
                    activity_regularizer=self._activity_regularizer,
                    kernel_constraint=self._kernel_constraint,
                    bias_constraint=self._bias_constraint,
                    name="cross_attention_output",
                    use_bias=self._use_bias,
                )

        self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
        self._attention_layer_norm = T5LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )

        self._intermediate_dense = dense_einsum.DenseEinsum(
            output_shape=self._intermediate_size,
            activation=self._intermediate_activation,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            # This layer is always float32 for numeric stability.
            dtype=tf.float32,
            name="intermediate",
            use_bias=self._use_bias,
        )

        self._output_dense = dense_einsum.DenseEinsum(
            output_shape=hidden_size,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="output",
            use_bias=self._use_bias,
        )
        self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

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
        base_config = super(TransformerT5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call_training(self, inputs, position_bias=None, cache_key=None, cache_value=None):
        """
        Training pipeline
        """
        input_tensor, attention_mask = inputs
        input_tensor_norm = self._pre_attention_norm(input_tensor)
        attention_inputs = [input_tensor_norm, input_tensor_norm]

        if attention_mask is not None:
            attention_inputs.append(attention_mask)

        attention_output, position_bias, key, value = self._attention_layer(
            attention_inputs,
            position_bias=position_bias,
            cache_key=cache_key,
            cache_value=cache_value,
        )

        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
        # Use float32 in keras layer norm and the gelu activation in the
        # intermediate dense layer for numeric stability
        if self.dtype == tf.float16:
            input_tensor = tf.cast(input_tensor, tf.float32)
            attention_output = tf.cast(attention_output, tf.float32)

        attention_output = input_tensor + attention_output
        attention_output_normed = self._attention_layer_norm(attention_output)
        intermediate_output = self._intermediate_dense(attention_output_normed)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        # During mixed precision training, attention_output is from layer norm and
        # is always fp32 for now. Cast layer_output to fp32 for the subsequent
        # add.
        # Use float32 in keras layer norm for numeric stability
        if self.dtype == tf.float16:
            layer_output = tf.cast(layer_output, tf.float32)
        # layer_output = self._output_layer_norm(layer_output + attention_output)
        return layer_output + attention_output, position_bias, key, value

    def call_decoder(
        self,
        inputs,
        position_bias=None,
        decoder_encoder_position_bias=None,
        cache_key=None,
        cache_value=None,
    ):
        """
        Training pipeline
        """
        input_tensor, attention_mask, encoder_output, decoder_encoder_mask = inputs

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
        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
        # Use float32 in keras layer norm and the gelu activation in the
        # intermediate dense layer for numeric stability
        if self.dtype == tf.float16:
            input_tensor = tf.cast(input_tensor, tf.float32)
            attention_output = tf.cast(attention_output, tf.float32)

        attention_output = input_tensor + attention_output
        if self._is_decoder:
            attention_output_copy = tf.identity(attention_output, name="attention_output_copy")
            attention_output_norm = self._pre_cross_attention_norm(attention_output_copy)
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
            # Use float32 in keras layer norm and the gelu activation in the
            # intermediate dense layer for numeric stability
            if self.dtype == tf.float16:
                attention_output_copy = tf.cast(attention_output_copy, tf.float32)
                attention_output = tf.cast(attention_output, tf.float32)

            attention_output = attention_output + attention_output_copy

        attention_output_normed = self._attention_layer_norm(attention_output)
        intermediate_output = self._intermediate_dense(attention_output_normed)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        # During mixed precision training, attention_output is from layer norm and
        # is always fp32 for now. Cast layer_output to fp32 for the subsequent
        # add.
        # Use float32 in keras layer norm for numeric stability
        if self.dtype == tf.float16:
            layer_output = tf.cast(layer_output, tf.float32)
        # layer_output = self._output_layer_norm(layer_output + attention_output)
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
        """
        Args:
            # Scalar dimensions referenced here:
            #   B = batch size (number of sequences)
            #   F = `from_tensor` sequence length
            #   T = `to_tensor` sequence length
            #   N = `num_attention_heads`
            #   H = `size_per_head`
            #   E = N * H
            # `query_tensor` = [B, F, N ,H]
            inputs: list, [input_embeddings, attention_mask]
                          [B x F x E, B x F x F]

        Returns:
            list: [layer_output, key, value]
                  if cache = None:
                    [B x E,      B x N x F x H , B x N x F x H]
                  else:
                    [B x E,      B x N x (F+T) x H , B x N x (F+T) x H]
                    (F+T) means , we will add the current (F) to cached key and value of length (T)

        """
        if self._is_decoder:
            outputs = self.call_decoder(
                inputs,
                position_bias=position_bias,
                decoder_encoder_position_bias=decoder_encoder_position_bias,
                cache_key=cache_key,
                cache_value=cache_value,
            )

        else:
            # outputs = self.call_training(inputs,
            #                             position_bias=position_bias,
            #                             decoder_encoder_position_bias=decoder_encoder_position_bias,
            #                             cache_key=cache_key,
            #                             cache_value=cache_value)
            outputs = self.call_training(
                inputs,
                position_bias=position_bias,
                cache_key=cache_key,
                cache_value=cache_value,
            )
        return outputs
