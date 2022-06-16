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
import math

import tensorflow as tf

from tf_transformers.layers import dense_einsum
from tf_transformers.layers.mask import CrossAttentionMask, masked_softmax
from tf_transformers.utils import tf_utils


class ImageSelfAttention(tf.keras.layers.Layer):
    """ImageSelfAttention layer.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-width vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-12,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        use_bias=True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name="attention",
        is_training=False,
        use_dropout=False,
        **kwargs,
    ):
        """
        Args:
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            kernel_initializer: Initializer for dense layer kernels.
            bias_initializer: Initializer for dense layer biases.
            kernel_regularizer: Regularizer for dense layer kernels.
            bias_regularizer: Regularizer for dense layer biases.
            activity_regularizer: Regularizer for dense layer activity.
            kernel_constraint: Constraint for dense layer kernels.
            bias_constraint: Constraint for dense layer kernels.
        """
        kwargs["name"] = name
        super(ImageSelfAttention, self).__init__(**kwargs)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        assert self._embed_dim % self._num_heads == 0
        head_size = self._embed_dim // self._num_heads
        self._head_size = head_size
        self._dropout_rate = dropout_rate
        self._layer_norm_epsilon = layer_norm_epsilon
        self._use_bias = use_bias
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.is_training = is_training
        self.use_dropout = use_dropout

        # We initially project all inputs to this dimension for
        # consistent attention values
        self._projection_dense = tf.keras.layers.Dense(
            embed_dim,
            use_bias=self._use_bias,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="projection",
        )

        self._query_dense = dense_einsum.DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="query",
        )

        self._key_dense = dense_einsum.DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="key",
        )

        self._value_dense = dense_einsum.DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="value",
        )
        # Self Attention Norm
        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )
        self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])
        self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    def build(self, input_shapes):

        image_shape, _ = input_shapes
        B, H, W, C = image_shape
        assert C % self._num_heads == 0

        # We use a projection back to original dimension
        # if C != self._embed_dim:
        self._projection_dense_back = tf.keras.layers.Dense(
            C,
            use_bias=self._use_bias,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="projection_back",
        )

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "dropout_rate": self._dropout_rate,
            "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self._bias_constraint),
        }
        base_config = super(ImageSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def merge_attention_heads(x):
        batch, n_heads, sequence, feature_length = tf_utils.get_shape_list(x)
        return tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [batch, sequence, n_heads * feature_length])

    def call(self, inputs):
        """
        inputs: [from_tensor(3D), to_tensor(3D)]

        No attention mask is required for image to image self attention
        """

        from_tensor = inputs[0]  # 4D
        to_tensor = inputs[1]  # 4D
        attention_mask = None

        B, H, W, C = from_tensor.shape
        hw = H * W
        assert C % self._num_heads == 0

        from_tensor = self._projection_dense(from_tensor)
        to_tensor = self._projection_dense(to_tensor)

        C_scaled = from_tensor.shape[-1]
        from_tensor_3d = tf.reshape(from_tensor, (-1, hw, C_scaled))
        to_tensor_3d = tf.reshape(to_tensor, (-1, hw, C_scaled))
        # Squeeze H, W into one
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query_tensor` = [B, F, N ,H]
        query_tensor = self._query_dense(from_tensor_3d)

        # `key_tensor` = [B, T, N, H]
        key_tensor = self._key_dense(to_tensor_3d)

        # `value_tensor` = [B, T, N, H]
        value_tensor = self._value_dense(to_tensor_3d)

        # Transpose to [B, N, T, H]
        query_tensor = tf.transpose(query_tensor, [0, 2, 1, 3])
        key_tensor = tf.transpose(key_tensor, [0, 2, 1, 3])
        value_tensor = tf.transpose(value_tensor, [0, 2, 1, 3])
        # attention_scores = tf.einsum(
        #      "BNFH,BNTH->BNFT",  query_tensor, key_tensor)
        attention_scores = tf.matmul(query_tensor, key_tensor, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))
        attention_probs = self._masked_softmax([attention_scores, attention_mask])
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self._dropout(attention_probs, training=self.use_dropout)
        # `context_layer` = [B, N, F, H]
        # context_layer = tf.einsum(
        #     "BNFT,BNTH->BNFH", attention_probs, value_tensor)

        context_layer = tf.matmul(attention_probs, value_tensor)

        context_layer_merged = self.merge_attention_heads(context_layer)
        context_layer_merged = tf.reshape(context_layer_merged, (-1, H, W, C_scaled))
        context_projected_back = self._projection_dense_back(context_layer_merged)
        context_layer_merged = self._attention_layer_norm(context_projected_back)

        return context_layer_merged, key_tensor, value_tensor


class ImageTextCrossAttention(tf.keras.layers.Layer):
    """ImageTextCrossAttention layer.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-width vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-12,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        use_bias=True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name="attention",
        is_training=False,
        use_dropout=False,
        **kwargs,
    ):
        """
        Args:
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            kernel_initializer: Initializer for dense layer kernels.
            bias_initializer: Initializer for dense layer biases.
            kernel_regularizer: Regularizer for dense layer kernels.
            bias_regularizer: Regularizer for dense layer biases.
            activity_regularizer: Regularizer for dense layer activity.
            kernel_constraint: Constraint for dense layer kernels.
            bias_constraint: Constraint for dense layer kernels.
        """
        kwargs["name"] = name
        super(ImageTextCrossAttention, self).__init__(**kwargs)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        assert self._embed_dim % self._num_heads == 0
        head_size = self._embed_dim // self._num_heads
        self._head_size = head_size
        self._dropout_rate = dropout_rate
        self._layer_norm_epsilon = layer_norm_epsilon
        self._use_bias = use_bias
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.is_training = is_training
        self.use_dropout = use_dropout

        # We initially project all inputs to this dimension for
        # consistent attention values
        self._projection_dense = tf.keras.layers.Dense(
            embed_dim,
            use_bias=self._use_bias,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="projection",
        )

        self._text_projection_dense = tf.keras.layers.Dense(
            embed_dim,
            use_bias=self._use_bias,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="text_projection",
        )

        self._query_dense = dense_einsum.DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="query",
        )

        self._key_dense = dense_einsum.DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="key",
        )

        self._value_dense = dense_einsum.DenseEinsum(
            output_shape=(self._num_heads, self._head_size),
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="value",
        )
        # Self Attention Norm
        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._layer_norm_epsilon,
            dtype=tf.float32,
        )
        self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])
        self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    def build(self, input_shapes):

        image_shape, text_emb_shape, _ = input_shapes
        B, H, W, C = image_shape
        assert C % self._num_heads == 0

        # We use a projection back to original dimension
        # if C != self._embed_dim:
        self._projection_dense_back = tf.keras.layers.Dense(
            C,
            use_bias=self._use_bias,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            name="projection_back",
        )

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "dropout_rate": self._dropout_rate,
            "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self._bias_constraint),
        }
        base_config = super(ImageTextCrossAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def merge_attention_heads(x):
        batch, n_heads, sequence, feature_length = tf_utils.get_shape_list(x)
        return tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [batch, sequence, n_heads * feature_length])

    def call(self, inputs):
        """
        inputs: [from_tensor(3D), to_tensor(3D)]

        No attention mask is required for image to image self attention
        """

        from_tensor = inputs[0]  # 4D
        to_tensor = inputs[1]  # 3D
        input_mask = inputs[2]  # 2D (batch_size x sequence_length)

        B, H, W, C = from_tensor.shape
        assert C % self._num_heads == 0

        from_tensor = self._projection_dense(from_tensor)
        to_tensor = self._text_projection_dense(to_tensor)

        B, H, W, C_scaled = from_tensor.shape
        hw = H * W
        from_tensor_3d = tf.reshape(from_tensor, (-1, hw, C_scaled))
        # to_tensor_3d = tf.reshape(to_tensor, (-1 , hw_to, C_scaled))
        # Concatanate image to_tensor to text to_tensor
        # to_tensor_3d = tf.concat([from_tensor_3d, to_tensor_3d], axis=1)
        to_tensor = tf.concat([from_tensor_3d, to_tensor], axis=1)
        # We need to take care of input mask
        hw = from_tensor_3d.shape[1]  # H * W
        image_mask = tf.ones((tf.shape(input_mask)[0], hw), dtype=input_mask.dtype)
        input_mask = tf.concat([image_mask, input_mask], axis=1)
        dummy_image_input_ids = tf.ones((tf.shape(input_mask)[0], hw), dtype=input_mask.dtype)
        attention_mask = CrossAttentionMask()([dummy_image_input_ids, input_mask])

        # Squeeze H, W into one
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query_tensor` = [B, F, N ,H]
        query_tensor = self._query_dense(from_tensor_3d)

        # `key_tensor` = [B, T, N, H]
        key_tensor = self._key_dense(to_tensor)

        # `value_tensor` = [B, T, N, H]
        value_tensor = self._value_dense(to_tensor)

        # Transpose to [B, N, T, H]
        query_tensor = tf.transpose(query_tensor, [0, 2, 1, 3])
        key_tensor = tf.transpose(key_tensor, [0, 2, 1, 3])
        value_tensor = tf.transpose(value_tensor, [0, 2, 1, 3])
        # attention_scores = tf.einsum(
        #      "BNFH,BNTH->BNFT",  query_tensor, key_tensor)

        attention_scores = tf.matmul(query_tensor, key_tensor, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))
        attention_probs = self._masked_softmax([attention_scores, attention_mask])
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self._dropout(attention_probs, training=self.use_dropout)
        # `context_layer` = [B, N, F, H]
        # context_layer = tf.einsum(
        #     "BNFT,BNTH->BNFH", attention_probs, value_tensor)

        context_layer = tf.matmul(attention_probs, value_tensor)
        context_layer_merged = self.merge_attention_heads(context_layer)
        context_layer_merged = tf.reshape(context_layer_merged, (-1, H, W, C_scaled))
        context_projected_back = self._projection_dense_back(context_layer_merged)
        context_layer_merged = self._attention_layer_norm(context_projected_back)

        return context_layer_merged, key_tensor, value_tensor
