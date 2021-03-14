# pylint: disable=g-classes-have-attributes
# from __future__ import google_type_annotations
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tf_transformers.core import LegacyLayer
from tf_transformers.layers import OnDeviceEmbedding, dense_einsum
from tf_transformers.layers.mask import masked_softmax
from tf_transformers.utils import tf_utils
from tf_transformers.utils.positional_bias_utils import compute_positional_bias


class T5Attention(LegacyLayer):
    """T5Attention layer.

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

    T5 Attention has a position_bias (relative positional bias) https://arxiv.org/abs/1803.02155
    only in first layer. Subsequent layer reuses it.

    """

    def __init__(
        self,
        num_heads,
        head_size,
        bidirectional,
        create_positonal_embedding=True,
        positional_buckets=32,
        dropout_rate=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_bias=True,
        name="attention",
        **kwargs,
    ):
        """
        Args:
            num_heads: Number of attention heads.
            head_size: Size of each attention head.
            bidirectional: bool, based on masking
            create_positonal_embedding: bool, to create positional embedding
                                        (T5 creates only it at layer1)
            positional_buckets: Positional buckets
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
        super(T5Attention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._head_size = head_size
        self._bidirectional = bidirectional
        self._create_positonal_embedding = create_positonal_embedding
        self._positional_buckets = positional_buckets
        self._dropout_rate = dropout_rate
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._use_bias = use_bias

        if self._create_positonal_embedding:
            # self._relative_embedding = tf.keras.layers.Embedding(
            #             self._positional_buckets, self._num_heads, name="relative_attention_bias")
            self._relative_embedding = OnDeviceEmbedding(
                self._positional_buckets,
                self._num_heads,
                name="relative_attention_bias",
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
            use_bias=use_bias,
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
            use_bias=use_bias,
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
            use_bias=use_bias,
        )

        self._masked_softmax = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])
        self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "head_size": self._head_size,
            "positional_buckets": self._positional_buckets,
            "dropout_rate": self._dropout_rate,
            "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self._bias_constraint),
            "use_bias": self._use_bias,
        }
        base_config = super(T5Attention, self).get_config()
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

    def call_predict(
        self,
        inputs,
        position_bias=None,
        cache_key=None,
        cache_value=None,
        cross_decoder_mode=True,
    ):
        from_tensor = inputs[0]
        to_tensor = inputs[1]
        attention_mask = inputs[2] if len(inputs) == 3 else None
        position_bias = position_bias
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query_tensor` = [B, F, N ,H]

        def left(position_bias, cross_decoder_mode):
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

            attention_scores = tf.einsum("BNFH,BNTH->BNFT", query_tensor, key_tensor)

            if cross_decoder_mode:
                position_bias = tf.zeros_like(attention_scores)
            else:
                if position_bias is None:
                    qlen = tf.shape(query_tensor)[2]
                    klen = tf.shape(key_tensor)[2]
                    rp_bucket = compute_positional_bias(
                        qlen,
                        klen,
                        bidirectional=self._bidirectional,
                        num_buckets=self._positional_buckets,
                    )
                    values = self._relative_embedding(rp_bucket)  # shape (qlen, klen, num_heads)
                    position_bias = tf.expand_dims(
                        tf.transpose(values, [2, 0, 1]), axis=0
                    )  # shape (1, num_heads, qlen, klen)
                    # To account for masking in postional buckets
                    # positon_mask = tf.expand_dims(tf.cast(tf.not_equal(attention_mask, 1.0),
                    #                                       position_bias.dtype) * -1e+09, 1)
                    # position_bias += positon_mask

            # T5 don't have this
            # attention_scores = tf.multiply(attention_scores,
            #                               1.0 / math.sqrt(float(self._head_size)))

            # Normalize the attention scores to probabilities.
            # `attention_probs` = [B, N, F, T]
            attention_scores_mask = tf.cast(tf.equal(attention_scores, 0.0), tf.float32) * -10000

            attention_scores += position_bias

            attention_scores += attention_scores_mask

            return (
                query_tensor,
                key_tensor,
                value_tensor,
                position_bias,
                attention_scores,
            )

        def right(position_bias, cross_decoder_mode):
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

            if cross_decoder_mode:
                attention_scores = tf.einsum("BNFH,BNTH->BNFT", query_tensor, key_tensor)
                if position_bias is None:
                    position_bias = tf.zeros_like(attention_scores)
                # T5 don't have this
                # attention_scores = tf.multiply(attention_scores,
                #                               1.0 / math.sqrt(float(self._head_size)))

                # Normalize the attention scores to probabilities.
                # `attention_probs` = [B, N, F, T]
                attention_scores_mask = tf.cast(tf.equal(attention_scores, 0.0), tf.float32) * -10000

                # We need only last position bias in the case of caching

                # attention_scores += tf.expand_dims(position_bias[:, :, -1, :], 2)

                attention_scores += attention_scores_mask
            else:
                key_tensor, value_tensor = self._update_cache(
                    key_tensor,
                    value_tensor,
                    cache_key,
                    cache_value,
                    decode_loop_step=None,
                )
                if position_bias is None:
                    # We need to add cache key length to account for previous word
                    qlen = tf.shape(query_tensor)[2] + tf.shape(cache_value)[2]
                    klen = tf.shape(key_tensor)[2]
                    rp_bucket = compute_positional_bias(
                        qlen,
                        klen,
                        bidirectional=self._bidirectional,
                        num_buckets=self._positional_buckets,
                    )
                    values = self._relative_embedding(rp_bucket)  # shape (qlen, klen, num_heads)
                    position_bias = tf.expand_dims(
                        tf.transpose(values, [2, 0, 1]), axis=0
                    )  # shape (1, num_heads, qlen, klen)
                    # To account for masking in postional buckets
                    # positon_mask = tf.expand_dims(tf.cast(tf.not_equal(attention_mask, 1.0),
                    #                                   position_bias.dtype) * -1e+09, 1)
                    # position_bias += positon_mask

                attention_scores = tf.einsum("BNFH,BNTH->BNFT", query_tensor, key_tensor)
                # T5 don't have this
                # attention_scores = tf.multiply(attention_scores,
                #                               1.0 / math.sqrt(float(self._head_size)))

                # Normalize the attention scores to probabilities.
                # `attention_probs` = [B, N, F, T]
                attention_scores_mask = tf.cast(tf.equal(attention_scores, 0.0), tf.float32) * -10000

                # We need only last position bias in the case of caching
                attention_scores += tf.expand_dims(position_bias[:, :, -1, :], 2)

                attention_scores += attention_scores_mask

            return (
                query_tensor,
                key_tensor,
                value_tensor,
                position_bias,
                attention_scores,
            )

        (query_tensor, key_tensor, value_tensor, position_bias, attention_scores,) = tf.cond(
            tf.equal(tf.reduce_sum(cache_key), 0.0),
            lambda: left(position_bias, cross_decoder_mode),
            lambda: right(position_bias, cross_decoder_mode),
        )

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        #   E = `embedding_dimension`

        attention_probs = self._masked_softmax([attention_scores, attention_mask])
        # Why multiply with this mask? When we have past key , in the case of variable batch
        # we need not to consider padding values for softmax. So this is the hack
        attention_probs = attention_probs * tf.expand_dims(attention_mask, 1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self._dropout(attention_probs, training=self.use_dropout)

        # `context_layer` = [B, N, F, H]
        context_layer = tf.einsum("BNFT,BNTH->BNFH", attention_probs, value_tensor)
        return (
            self.merge_attention_heads(context_layer),
            position_bias,
            key_tensor,
            value_tensor,
        )

    def call_training(self, inputs, position_bias=None):
        """
        inputs: [from_tensor(3D), to_tensor(3D), attention_mask(3D),
                positional_bias[None, 3D(batch, qlen, vlen)]]
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

        attention_scores = tf.matmul(query_tensor, key_tensor, transpose_b=True)

        #         T5 don't have this
        #         attention_scores = tf.multiply(attention_scores,
        #                                        1.0 / math.sqrt(float(self._head_size)))
        if self._create_positonal_embedding:
            if position_bias is None:
                qlen = tf.shape(query_tensor)[2]
                klen = tf.shape(key_tensor)[2]
                rp_bucket = compute_positional_bias(
                    qlen,
                    klen,
                    bidirectional=self._bidirectional,
                    num_buckets=self._positional_buckets,
                )
                values = self._relative_embedding(rp_bucket)  # shape (qlen, klen, num_heads)
                position_bias = tf.expand_dims(
                    tf.transpose(values, [2, 0, 1]), axis=0
                )  # shape (1, num_heads, qlen, klen)
                # To account for masking in postional buckets
                positon_mask = tf.expand_dims(
                    tf.cast(tf.not_equal(attention_mask, 1.0), position_bias.dtype) * -1e09,
                    1,
                )
                position_bias += positon_mask
            attention_scores += position_bias
        else:
            # T5 cross attention decoder has no relative bias positional
            # embedding. I have code it with that it mind, so tf.zeros_like
            # is a hack
            position_bias = tf.zeros_like(attention_scores)
            attention_scores += position_bias

        attention_probs = self._masked_softmax([attention_scores, attention_mask])
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self._dropout(attention_probs, training=self.use_dropout)
        # `context_layer` = [B, N, F, H]
        # context_layer = tf.einsum(
        #     "BNFT,BNTH->BNFH", attention_probs, value_tensor)

        context_layer = tf.matmul(attention_probs, value_tensor)

        return (
            self.merge_attention_heads(context_layer),
            position_bias,
            key_tensor,
            value_tensor,
        )

    def call(
        self,
        inputs,
        position_bias=None,
        cache_key=None,
        cache_value=None,
        cross_decoder_mode=False,
    ):
        if self.is_training:
            (
                attention_states,
                position_bias,
                key_tensor,
                value_tensor,
            ) = self.call_training(inputs, position_bias)
            return attention_states, position_bias, key_tensor, value_tensor
        else:
            (
                attention_states,
                position_bias,
                key_tensor,
                value_tensor,
            ) = self.call_predict(inputs, position_bias, cache_key, cache_value, cross_decoder_mode)
            return attention_states, position_bias, key_tensor, value_tensor
