import tensorflow as tf


class TimeEmbedding(tf.keras.layers.Layer):
    """Creates a sinusoidal embedding.

    This layer creates a sinusoidal embedding as described in "Attention is All you need":

    """

    def __init__(
        self,
        n_channels,
        initializer="glorot_uniform",
        bias_initializer='zeros',
        name="time_embeddings",
        activation='swish',
        use_bias=True,
        dtype=tf.float32,
        **kwargs,
    ):
        """
        Args:
            n_channels ([int]): Similar to embedding size
            scale_factor ([int]): How much to scale embedding size for next dense layer
            initializer (str, optional): The initializer to use for the
            embedding weights. Defaults to "glorot_uniform".
            name (str, optional): name of the layer. Defaults to "positional_embeddings".
            dtype ([type], optional): [description]. Defaults to tf.float32.
        """
        super(TimeEmbedding, self).__init__(name=name, dtype=dtype, **kwargs)

        assert (n_channels % 2) == 0
        self._n_channels = n_channels
        self._initializer = initializer
        self._bias_initializer = bias_initializer
        self._dtype = dtype

    def get_config(self):
        """Config based on init arguments

        Returns:
            [dict]: Dict of all init arguments
        """
        config = {
            "n_channels": self._n_channels,
            "initializer": self._initializer,
            "name": self._name,
            "dtype": self._dtype,
        }
        base_config = super(TimeEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, timesteps):
        """Call

        Args:
            timesteps ([tf.Tensor]): input ids 1D timesteps (B, ) B is batch_size
            eg: <tf.Tensor: shape=(4,), dtype=int32, numpy=array([719, 178, 485, 431], dtype=int32)>

        Returns:
            [tf.Tensor]: embeddings 3D (b x s x h)
        """
        half_dim = self._n_channels // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=self._dtype) * -emb)  # 1-D vector of size half_dim
        emb = tf.cast(tf.expand_dims(timesteps, axis=1), self._dtype) * tf.expand_dims(emb, axis=0)  # B x half_dim
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)  # B x self._n_channels

        return emb
