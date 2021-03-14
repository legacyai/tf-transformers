# pylint: disable=g-classes-have-attributes
# from __future__ import google_type_annotations
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_transformers.core import LegacyLayer
from tf_transformers.layers import dense_einsum
from tf_transformers.layers.attention import BigBirdAttention, BlockMultiHeadAttention, MultiHeadAttention


class TransformerBERT(LegacyLayer):
    """Transformer

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
        use_bias=True,
        is_decoder=None,
        share_attention_layers=True,
        layer_norm_epsilon=None,
        cross_attention_inside_encoder=False,
        attention_type="full_attention",  # ['full_attention', 'block_attention']
        block_size=64,
        num_rand_blocks=3,
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
        kwargs["name"] = name
        super(TransformerBERT, self).__init__(**kwargs)
        self._attention_cfg = attention_cfg
        self._num_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = intermediate_activation
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
        self._cross_attention_inside_encoder = cross_attention_inside_encoder
        self._attention_type = attention_type
        self._attention_head_size = attention_head_size
        self._num_rand_blocks = num_rand_blocks
        self._from_block_size = self._to_block_size = block_size

        if self._is_decoder is None:
            raise ValueError(
                "`is_decoder` should be set to bool. You have \
                to set it where you call the transformer layers"
            )

    def build(self, input_shape):
        """
        Args:
            input_shape: [word_embeddings (3D), attention_mask(3D)]
        """
        input_tensor = input_shape[0]
        input_tensor_shape = tf.TensorShape(input_tensor)
        if len(input_tensor_shape) != 3:
            raise ValueError("TransformerBERT expects a three-dimensional input of " "shape [batch, sequence, width].")
        batch_size, sequence_length, hidden_size = input_tensor_shape

        if not self._attention_head_size:
            if hidden_size % self._num_heads != 0:
                raise ValueError(
                    "The input size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, self._num_heads)
                )
            self._attention_head_size = int(hidden_size // self._num_heads)

        if self._attention_type == "full_attention":

            self._attention_layer = MultiHeadAttention(
                num_heads=self._num_heads,
                head_size=self._attention_head_size,
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
            )

        if self._attention_type == "block_attention":

            self._attention_layer = BlockMultiHeadAttention(
                num_heads=self._num_heads,
                head_size=self._attention_head_size,
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
            )

        if self._attention_type == "bigbird":
            self._attention_layer = BigBirdAttention(
                num_heads=self._num_heads,
                head_size=self._attention_head_size,
                num_rand_blocks=self._num_rand_blocks,
                from_block_size=self._from_block_size,
                to_block_size=self._to_block_size,
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
        )

        self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )

        # If we have cross attention inside encoder
        if self._cross_attention_inside_encoder:

            # Hard setting is_training to True, as we do not have to use cache here
            self._cross_attention_layer = MultiHeadAttention(
                num_heads=self._num_heads,
                head_size=self._attention_head_size,
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
            )

            self._cross_attention_layer_norm = tf.keras.layers.LayerNormalization(
                name="cross_attention_layer_norm",
                axis=-1,
                epsilon=self._layer_norm_epsilon,
                dtype=tf.float32,
            )

        if self._is_decoder:
            if self._share_attention_layers:
                self._cross_attention_layer = self._attention_layer
                self._cross_attention_output_dense = self._attention_output_dense
                self._cross_attention_layer_norm = self._attention_layer_norm

            else:
                # Hard setting is_training to True, as we do not have to use cache here
                self._cross_attention_layer = MultiHeadAttention(
                    num_heads=self._num_heads,
                    head_size=self._attention_head_size,
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
                )

                self._cross_attention_layer_norm = tf.keras.layers.LayerNormalization(
                    name="cross_attention_layer_norm",
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
        )
        self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

        # Use float32 in layernorm for numeric stability.
        self._output_layer_norm = tf.keras.layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )
        super(TransformerBERT, self).build(input_shape)

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
        base_config = super(TransformerBERT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call_training(self, inputs, cache_key=None, cache_value=None):
        """
        Training pipeline
        """
        input_tensor, attention_mask = inputs

        # For bigbird (attention_mask is input_mask) 2D
        # No pre norm for GPT2
        # input_tensor_norm = self._pre_attention_norm(input_tensor)
        attention_inputs = [input_tensor, input_tensor]

        if attention_mask is not None:
            attention_inputs.append(attention_mask)
        attention_output, key, value = self._attention_layer(
            attention_inputs, cache_key=cache_key, cache_value=cache_value
        )
        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
        # Use float32 in keras layer norm and the gelu activation in the
        # intermediate dense layer for numeric stability
        if self.dtype == tf.float16:
            input_tensor = tf.cast(input_tensor, tf.float32)
            attention_output = tf.cast(attention_output, tf.float32)

        attention_output = self._attention_layer_norm(input_tensor + attention_output)
        intermediate_output = self._intermediate_dense(attention_output)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        # During mixed precision training, attention_output is from layer norm and
        # is always fp32 for now. Cast layer_output to fp32 for the subsequent
        # add.
        # Use float32 in keras layer norm for numeric stability
        if self.dtype == tf.float16:
            layer_output = tf.cast(layer_output, tf.float32)
            attention_output = tf.cast(attention_output, dtype=tf.float32)
        layer_output = self._output_layer_norm(tf.cast(layer_output, dtype=tf.float32) + attention_output)
        return layer_output, key, value

    def call_cross_attention_in_encoder(self, inputs, mode, cache_key=None, cache_value=None):
        """
        inputs: dict keys:
                if self.is_training:
                    encoder_input_ids, decoder_input_ids
                    optional -> encoder_input_mask, decoder_input_mask, encoder_input_type_ids \
                        decoder_input_type_ids,
                else:

        Training pipeline
        """

        def calculate_self_attention(input_tensor, attention_mask):
            # Encoder
            # input_tensor_norm = self._pre_attention_norm(input_tensor)
            attention_inputs = [input_tensor, input_tensor]

            if attention_mask is not None:
                attention_inputs.append(attention_mask)

            # Do not worry about cache_key
            # When is_training = True, it will have value
            # and compute update_cache in self attention
            attention_output, key, value = self._attention_layer(
                attention_inputs, cache_key=cache_key, cache_value=cache_value
            )
            attention_output = self._attention_output_dense(attention_output)
            attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
            # Use float32 in keras layer norm and the gelu activation in the
            # intermediate dense layer for numeric stability
            if self.dtype == tf.float16:
                input_tensor = tf.cast(input_tensor, tf.float32)
                attention_output = tf.cast(attention_output, tf.float32)
            attention_output = self._attention_layer_norm(input_tensor + attention_output)
            return attention_output, key, value

        (
            input_tensor,
            attention_mask,
            decoder_encoder_mask,  # None when  mode == 'encoder'
            encoder_attention_output,  # None when mode == 'encoder'
        ) = inputs
        if mode == "encoder":
            encoder_attention_output, _, _ = calculate_self_attention(input_tensor, attention_mask)
            return encoder_attention_output, _, _
        if mode == "decoder":
            (
                decoder_attention_output,
                key,
                value,
            ) = calculate_self_attention(input_tensor, attention_mask)

        decoder_cross_attention_inputs = [
            decoder_attention_output,
            encoder_attention_output,
            decoder_encoder_mask,
        ]
        # When is_training is True
        if cache_key is not None:
            cache_key_temp = tf.zeros_like(cache_key)
            cache_value_temp = tf.zeros_like(cache_value)
        else:
            cache_key_temp = None
            cache_value_temp = None

        cross_attention_output, _, _ = self._cross_attention_layer(
            decoder_cross_attention_inputs, cache_key=cache_key_temp, cache_value=cache_value_temp
        )

        cross_attention_output = self._cross_attention_output_dense(cross_attention_output)
        cross_attention_output = self._attention_dropout(cross_attention_output, training=self.use_dropout)
        cross_attention_output = self._cross_attention_layer_norm(decoder_attention_output + cross_attention_output)

        # Use float32 in keras layer norm and the gelu activation in the
        # intermediate dense layer for numeric stability
        if self.dtype == tf.float16:
            decoder_attention_output = tf.cast(decoder_attention_output, tf.float32)
            cross_attention_output = tf.cast(cross_attention_output, tf.float32)

        intermediate_output = self._intermediate_dense(cross_attention_output)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        # During mixed precision training, attention_output is from layer norm and
        # is always fp32 for now. Cast layer_output to fp32 for the subsequent
        # add.
        # Use float32 in keras layer norm for numeric stability
        if self.dtype == tf.float16:
            layer_output = tf.cast(layer_output, tf.float32)
        layer_output = self._output_layer_norm(layer_output + cross_attention_output)
        return layer_output, key, value

    def call_decoder(self, inputs, cache_key=None, cache_value=None):
        """
        Training pipeline
        """
        input_tensor, attention_mask, encoder_output, decoder_encoder_mask = inputs

        attention_inputs = [input_tensor, input_tensor]

        if attention_mask is not None:
            attention_inputs.append(attention_mask)
        attention_output, key, value = self._attention_layer(
            attention_inputs, cache_key=cache_key, cache_value=cache_value
        )

        attention_output = self._attention_output_dense(attention_output)
        attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
        attention_output = self._attention_layer_norm(attention_output + input_tensor)
        # Use float32 in keras layer norm and the gelu activation in the
        # intermediate dense layer for numeric stability
        if self.dtype == tf.float16:
            input_tensor = tf.cast(input_tensor, tf.float32)
            attention_output = tf.cast(attention_output, tf.float32)

        if self._is_decoder:
            attention_output_copy = tf.identity(attention_output, name="attention_output_copy")
            attention_inputs_for_decoder = [
                attention_output_copy,
                encoder_output,
                decoder_encoder_mask,
            ]

            # This is required, if we share the parameters
            if cache_key is not None:
                cache_key_cross = tf.zeros_like(cache_key)
                cache_value_cross = tf.zeros_like(cache_value)
            else:
                cache_key_cross = None
                cache_value_cross = None

            attention_output, _, _ = self._cross_attention_layer(
                attention_inputs_for_decoder, cache_key=cache_key_cross, cache_value=cache_value_cross
            )

            attention_output = self._cross_attention_output_dense(attention_output)
            attention_output = self._attention_dropout(attention_output, training=self.use_dropout)
            attention_output = self._cross_attention_layer_norm(attention_output_copy + attention_output)
            # Use float32 in keras layer norm and the gelu activation in the
            # intermediate dense layer for numeric stability
            if self.dtype == tf.float16:
                attention_output_copy = tf.cast(attention_output_copy, tf.float32)
                attention_output = tf.cast(attention_output, tf.float32)
        intermediate_output = self._intermediate_dense(attention_output)
        layer_output = self._output_dense(intermediate_output)
        layer_output = self._output_dropout(layer_output)
        # During mixed precision training, attention_output is from layer norm and
        # is always fp32 for now. Cast layer_output to fp32 for the subsequent
        # add.
        # Use float32 in keras layer norm for numeric stability
        if self.dtype == tf.float16:
            layer_output = tf.cast(layer_output, tf.float32)
        layer_output = self._output_layer_norm(layer_output + attention_output)

        return layer_output, key, value

    def call(self, inputs, mode="encoder", cache_key=None, cache_value=None):
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
            outputs = self.call_decoder(inputs, cache_key=cache_key, cache_value=cache_value)

        elif self._cross_attention_inside_encoder:
            outputs = self.call_cross_attention_in_encoder(
                inputs, mode=mode, cache_key=cache_key, cache_value=cache_value
            )
        else:
            outputs = self.call_training(inputs, cache_key=cache_key, cache_value=cache_value)
        return outputs
