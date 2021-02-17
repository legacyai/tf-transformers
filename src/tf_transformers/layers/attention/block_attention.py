# from __future__ import google_type_annotations
from __future__ import absolute_import, division, print_function

import math

import numpy as np
import tensorflow as tf

from tf_transformers.core import LegacyLayer
from tf_transformers.layers import dense_einsum
from tf_transformers.layers.mask import masked_softmax
from tf_transformers.utils import tf_utils
from tf_transformers.layers.mask import SelfAttentionMask


# Lets HARDCODE few things
max_allowed_sequence_length = 512
to_block_size = 64

from_seq_length = 4096
from_block_size = 64
to_seq_length = 4096
to_block_size = 64


def get_qk_index_pos(n_rows, n_columns):
    """We generate random psitions to attend per block

    Args:
        n_rows ([type]): [description]
        n_columns ([type]): [description]

    Returns:
        [type]: [description]
    """
    qk_index_pos = []
    ix_size = max_allowed_sequence_length // to_block_size
    for i in range(n_rows):
        a = np.zeros(n_columns)
        ix_size = max_allowed_sequence_length // to_block_size
        ix = np.random.choice(len(a), size=ix_size, replace=False)
        a[ix] = 1.0
        qk_index_pos.append(a)
    return qk_index_pos


class BlockMultiHeadAttention(LegacyLayer):
    """BlockMultiHeadAttention layer.

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
        super(BlockMultiHeadAttention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._head_size = head_size
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
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def block_wise_attention_scores(self, query_tensor, key_tensor, attention_mask):
        """"""

        batch_size = tf.shape(query_tensor)[0]
        query_blocks = tf.split(query_tensor, axis=2, num_or_size_splits=from_seq_length // from_block_size)
        key_blocks = tf.split(key_tensor, axis=2, num_or_size_splits=to_seq_length // to_block_size)

        qk_index_pos = get_qk_index_pos(len(query_blocks), len(key_blocks))
        all_blocks = []
        attention_mask_q_split = tf.split(attention_mask, axis=1, num_or_size_splits=len(query_blocks))
        for q_index, q_block in enumerate(query_blocks):
            block_output = []
            non_zero_block_output = []
            attention_mask_block = []
            zero_block_output = []
            attention_mask_k_split = tf.split(
                attention_mask_q_split[q_index], axis=2, num_or_size_splits=len(key_blocks)
            )
            zero_tensor = tf.zeros(
                (batch_size, self._num_heads, from_seq_length // from_block_size, to_seq_length // to_block_size)
            )
            for k_index, k_block in enumerate(key_blocks):

                if qk_index_pos[q_index][k_index] == 1:
                    qk_block = tf.matmul(q_block, k_block, transpose_b=True)
                    non_zero_block_output.append(qk_block)
                    attention_mask_block.append(attention_mask_k_split[k_index])

                block_output.append(zero_tensor)

            non_zero_block_output_softmax_masked = self._masked_softmax(
                [tf.concat(non_zero_block_output, axis=-1), tf.concat(attention_mask_block, axis=2)]
            )
            non_zero_block_output_softmax_blocks = tf.split(
                non_zero_block_output_softmax_masked,
                axis=-1,
                num_or_size_splits=max_allowed_sequence_length // to_block_size,
            )

            non_zero_counter = 0
            for _index, _value in enumerate(qk_index_pos[q_index]):
                if _value == 1:
                    block_output[_index] = non_zero_block_output_softmax_blocks[non_zero_counter]
                    non_zero_counter += 1
            all_blocks.append(tf.concat(block_output, axis=-1))
        attention_probs = tf.concat(all_blocks, axis=2)
        return qk_index_pos, attention_probs

    def block_wise_full_calculations(self, query_tensor, key_tensor, value_tensor, input_mask):
        """Entire end to end attention and context cacluation happens here"""

        batch_size = tf.shape(query_tensor)[0]
        query_blocks = tf.split(query_tensor, axis=2, num_or_size_splits=from_seq_length // from_block_size)
        key_blocks = tf.split(key_tensor, axis=2, num_or_size_splits=to_seq_length // to_block_size)
        value_blocks = tf.split(value_tensor, axis=2, num_or_size_splits=to_seq_length // to_block_size)
        qk_index_pos = get_qk_index_pos(len(query_blocks), len(key_blocks))

        all_blocks = []
        input_mask_split = tf.split(input_mask, axis=1, num_or_size_splits=len(query_blocks))

        for q_index, q_block in enumerate(query_blocks):
            block_output = []
            non_zero_block_output = []
            attention_mask_block = []
            zero_block_output = []
            input_mask_block = []
            value_blocks_local = []
            zero_tensor = tf.zeros(
                (batch_size, self._num_heads, from_seq_length // from_block_size, to_seq_length // to_block_size)
            )
            for k_index, k_block in enumerate(key_blocks):

                if qk_index_pos[q_index][k_index] == 1:
                    qk_block = tf.matmul(q_block, k_block, transpose_b=True)
                    non_zero_block_output.append(qk_block)
                    input_mask_block.append(input_mask_split[k_index])
                    value_blocks_local.append(value_blocks[k_index])

            input_mask_block = tf.concat(input_mask_block, axis=1)

            local_attention_mask = SelfAttentionMask()(
                [tf.random.uniform(shape=(batch_size, to_block_size, 1)), input_mask_block]
            )
            attention_probs_local = self._masked_softmax(
                [tf.concat(non_zero_block_output, axis=-1), local_attention_mask]
            )

            value_blocks_local = tf.concat(value_blocks_local, axis=2)
            context_layer_local = tf.matmul(attention_probs_local, value_blocks_local)
            all_blocks.append(context_layer_local)

        return tf.concat(all_blocks, axis=2)

    def block_wise_context_caculation(self, qk_index_pos, attention_probs, value_tensor):
        """"""
        batch_size = tf.shape(value_tensor)[0]
        attention_probs_q_split = tf.split(attention_probs, axis=2, num_or_size_splits=from_seq_length // to_block_size)
        value_blocks = tf.split(value_tensor, axis=2, num_or_size_splits=to_seq_length // to_block_size)
        all_blocks2 = []
        for a_index, a_block in enumerate(attention_probs_q_split):
            block_output = 0
            attention_probs_k_split = tf.split(
                attention_probs_q_split[a_index], axis=3, num_or_size_splits=len(attention_probs_q_split)
            )
            zero_tensor = tf.zeros((batch_size, self._num_heads, from_seq_length // from_block_size, self._head_size))
            for v_index, v_block in enumerate(value_blocks):

                if qk_index_pos[a_index][v_index] == 1:
                    av_block = tf.matmul(attention_probs_k_split[v_index], v_block)
                    block_output += av_block
                else:
                    block_output += zero_tensor
            all_blocks2.append(block_output)
        context_layer = tf.concat(all_blocks2, axis=2)
        return context_layer

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

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        #   E = `embedding_dimension`

        attention_scores = tf.einsum("BNFH,BNTH->BNFT", query_tensor, key_tensor)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))
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

        # `key_tensor` = [B, T, N, H]
        key_tensor = self._key_dense(to_tensor)

        # `value_tensor` = [B, T, N, H]
        value_tensor = self._value_dense(to_tensor)

        # Transpose to [B, N, T, H]
        query_tensor = tf.transpose(query_tensor, [0, 2, 1, 3])
        key_tensor = tf.transpose(key_tensor, [0, 2, 1, 3])
        value_tensor = tf.transpose(value_tensor, [0, 2, 1, 3])

        # attention_scores = tf.einsum(
        #     "BNFH,BNTH->BNFT",  query_tensor, key_tensor)

        # attention_scores = tf.matmul(query_tensor, key_tensor, transpose_b=True)
        # attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._head_size)))
        # attention_probs = self._masked_softmax([attention_scores, attention_mask])
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self._dropout(attention_probs, training=self.use_dropout)
        # `context_layer` = [B, N, F, H]
        # context_layer = tf.einsum(
        #     "BNFT,BNTH->BNFH", attention_probs, value_tensor)

        # context_layer = tf.matmul(attention_probs, value_tensor)

        # qk_index_pos, attention_probs = self.block_wise_attention_scores(query_tensor, key_tensor, attention_mask)
        # context_layer = self.block_wise_context_caculation(qk_index_pos, attention_probs, value_tensor)

        context_layer = self.block_wise_full_calculations(query_tensor, key_tensor, value_tensor, attention_mask)
        return self.merge_attention_heads(context_layer), key_tensor, value_tensor

    def call(self, inputs, cache_key=None, cache_value=None):
        if self.is_training:
            attention_states, key_tensor, value_tensor = self.call_training(inputs)
            return attention_states, key_tensor, value_tensor
        else:
            attention_states, key_tensor, value_tensor = self.call_predict(inputs, cache_key, cache_value)
            return attention_states, key_tensor, value_tensor
