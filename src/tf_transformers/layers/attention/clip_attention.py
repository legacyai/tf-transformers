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

import tensorflow as tf

from tf_transformers.core import LegacyLayer
from tf_transformers.layers import dense_einsum
from tf_transformers.layers.mask import masked_softmax
from tf_transformers.utils import tf_utils


class CLIPMultiHeadAttention(LegacyLayer):
    """MultiHeadAttention layer.

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
        num_heads,
        head_size,
        is_training,
        use_auto_regressive,
        use_decoder=False,
        dropout_rate=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name="attention",
        **kwargs,
    ):
        """
        Args:
            num_heads: Number of attention heads.
            head_size: Size of each attention head.
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
        super(CLIPMultiHeadAttention, self).__init__(is_training=is_training, **kwargs)
        self._num_heads = num_heads
        self._head_size = head_size
        self._scale = self._head_size ** -0.5
        self._is_training = is_training
        self._use_auto_regressive = use_auto_regressive
        self._use_decoder = use_decoder
        self._dropout_rate = dropout_rate
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
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

        self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])
        self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "head_size": self._head_size,
            "dropout_rate": self._dropout_rate,
            "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self._bias_constraint),
        }
        base_config = super(CLIPMultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def merge_attention_heads(x):
        batch, n_heads, sequence, feature_length = tf_utils.get_shape_list(x)
        return tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [batch, sequence, n_heads * feature_length])

    def _update_cache(self, key_tensor, value_tensor, cache_key, cache_value, decode_loop_step):
        """Updates cache states and gets full-length key/value tensors."""
        # Combines cached keys and values with new keys and values.
        if decode_loop_step is not None:
            # TPU special case.
            key_seq_dim = cache_key.shape.as_list()[2]
            indices = tf.reshape(
                tf.one_hot(decode_loop_step, key_seq_dim, dtype=key_tensor.dtype),
                [1, 1, key_seq_dim, 1],
            )
            key_tensor = cache_key + key_tensor * indices
            value_seq_dim = cache_value.shape.as_list()[2]
            indices = tf.reshape(
                tf.one_hot(decode_loop_step, value_seq_dim, dtype=value_tensor.dtype),
                [1, 1, value_seq_dim, 1],
            )
            value_tensor = cache_value + value_tensor * indices
        else:
            key_tensor = tf.concat([tf.cast(cache_key, key_tensor.dtype), key_tensor], axis=2)
            value_tensor = tf.concat([tf.cast(cache_value, value_tensor.dtype), value_tensor], axis=2)

        # Update cache
        cache_key = key_tensor
        cache_value = value_tensor

        return key_tensor, value_tensor

    def call_predict(self, inputs, cache_key=None, cache_value=None):
        from_tensor = inputs[0]
        to_tensor = inputs[1]
        attention_mask = inputs[2] if len(inputs) == 3 else None
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query_tensor` = [B, F, N ,H]

        def left():
            query_tensor = self._query_dense(from_tensor)

            # `key_tensor` = [B, T, N, H]
            key_tensor = self._key_dense(to_tensor)

            # `value_tensor` = [B, T, N, H]
            value_tensor = self._value_dense(to_tensor)

            # Take the dot product between "query" and "key" to get the raw
            # attention scores.

            # `query_tensor` = [B, N, F, H]
            # 'key_tensor'   = [B, N, T, H]
            # `value_tensor` = [B, N, T, H]

            # Transpose to [B, N, T, H]
            query_tensor = tf.transpose(query_tensor, [0, 2, 1, 3])
            key_tensor = tf.transpose(key_tensor, [0, 2, 1, 3])
            value_tensor = tf.transpose(value_tensor, [0, 2, 1, 3])

            return query_tensor, key_tensor, value_tensor

        def right():
            query_tensor = self._query_dense(from_tensor)

            # `key_tensor` = [B, T, N, H]
            key_tensor = self._key_dense(to_tensor)

            # `value_tensor` = [B, T, N, H]
            value_tensor = self._value_dense(to_tensor)

            # Take the dot product between "query" and "key" to get the raw
            # attention scores.

            # `query_tensor` = [B, N, F, H]
            # 'key_tensor'   = [B, N, T, H]
            # `value_tensor` = [B, N, T, H]

            # Transpose to [B, N, T, H]
            query_tensor = tf.transpose(query_tensor, [0, 2, 1, 3])
            key_tensor = tf.transpose(key_tensor, [0, 2, 1, 3])
            value_tensor = tf.transpose(value_tensor, [0, 2, 1, 3])
            key_tensor, value_tensor = self._update_cache(
                key_tensor, value_tensor, cache_key, cache_value, decode_loop_step=None
            )
            return query_tensor, key_tensor, value_tensor

        query_tensor, key_tensor, value_tensor = tf.cond(
            tf.equal(tf.reduce_sum(cache_key), 0.0), lambda: left(), lambda: right()
        )
        query_tensor = query_tensor * self._scale
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        #   E = `embedding_dimension`

        attention_scores = tf.einsum("BNFH,BNTH->BNFT", query_tensor, key_tensor)
        # attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))
        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_scores_mask = tf.cast(tf.equal(attention_scores, 0.0), tf.float32) * -10000
        attention_scores += attention_scores_mask
        attention_probs = self._masked_softmax([attention_scores, attention_mask])
        # Why multiply with this mask? When we have past key , in the case of variable batch
        # we need not to consider padding values for softmax. So this is the hack
        attention_probs = attention_probs * tf.expand_dims(attention_mask, 1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self._dropout(attention_probs, training=self.use_dropout)

        # `context_layer` = [B, N, F, H]
        # context_layer = tf.einsum("BNFT,BNTH->BNFH", attention_probs, value_tensor)
        context_layer = tf.matmul(attention_probs, value_tensor)
        return self.merge_attention_heads(context_layer), key_tensor, value_tensor

    def call_training(self, inputs):
        """
        inputs: [from_tensor(3D), to_tensor(3D), attention_mask(3D)]
        """
        from_tensor = inputs[0]
        to_tensor = inputs[1]
        attention_mask = inputs[2] if len(inputs) == 3 else None
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query_tensor` = [B, F, N ,H]
        query_tensor = self._query_dense(from_tensor)
        query_tensor = query_tensor * self._scale
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
        # attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))
        attention_probs = self._masked_softmax([attention_scores, attention_mask])
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self._dropout(attention_probs, training=self.use_dropout)
        # `context_layer` = [B, N, F, H]
        # context_layer = tf.einsum(
        #     "BNFT,BNTH->BNFH", attention_probs, value_tensor)

        context_layer = tf.matmul(attention_probs, value_tensor)
        return self.merge_attention_heads(context_layer), key_tensor, value_tensor

    def call(self, inputs, cache_key=None, cache_value=None):
        # For decoder this function is used for training and inference
        if (self._is_training is False and self._use_auto_regressive) or cache_key is not None:
            attention_states, key_tensor, value_tensor = self.call_predict(inputs, cache_key, cache_value)
            return attention_states, key_tensor, value_tensor
        else:
            attention_states, key_tensor, value_tensor = self.call_training(inputs)
            return attention_states, key_tensor, value_tensor
