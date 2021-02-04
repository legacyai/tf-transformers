import tensorflow as tf

from tf_transformers.activations import get_activation


class MLMLayer(tf.keras.layers.Layer):
    """MLMLayer layer, which will consume the last_token_logits."""

    def __init__(self, embedding_size, layer_norm_epsilon, hidden_act, **kwargs):
        super(MLMLayer, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dense1 = tf.keras.layers.Dense(embedding_size)
        self.act = get_activation(hidden_act)
        self._extra_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=layer_norm_epsilon, dtype=tf.float32)

    def call(self, last_token_logits):
        """
        Args:
            last_token_logits: batch_size x embedding_size

        """
        intermediate_projection = self.act(self.dense1(last_token_logits))
        intermediate_projection_norm = self._extra_norm(intermediate_projection)
        return intermediate_projection_norm

    def get_config(self):
        config = {
            "embedding_size": self.embedding_size,
            "epsilon": self.layer_norm_epsilon,
            "activation": tf.keras.activations.serialize(self.act),
        }
        base_config = super(MLMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
