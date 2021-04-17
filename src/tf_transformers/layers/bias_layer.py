import tensorflow as tf


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, name="bias", trainable=True, initializer="zeros", *args, **kwargs):
        self._trainable = trainable
        self._initializer = initializer
        self._name = name
        super(BiasLayer, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias", shape=(input_shape[-1],), initializer=self._initializer, trainable=self._trainable
        )

    def call(self, x):
        return x + self.bias
