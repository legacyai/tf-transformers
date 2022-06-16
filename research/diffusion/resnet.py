import tensorflow as tf
import tensorflow_addons as tfa

from tf_transformers.activations import get_activation


class ResNetBlock(tf.keras.layers.Layer):
    """Creates a Resnet Block with Residual connection with Conv2D layers"""

    def __init__(
        self,
        out_ch,
        use_scale_shift_norm=True,
        kernel_size=(3, 3),
        strides=(1, 1),
        initializer="glorot_uniform",
        bias_initializer='zeros',
        name="resnet_block",
        activation='swish',
        use_bias=True,
        dtype=tf.float32,
        dropout_rate=0.0,
        use_dropout=False,
        **kwargs,
    ):
        """
        Args:
            out_ch (int): dimension of output channel
            kernel_size: Kernel size
            strides: Strides
            activation: Activation
            use_bias: To use bias or not
            initializer (str, optional): The initializer to use for the
            embedding weights. Defaults to "glorot_uniform".
            name (str, optional): name of the layer. Defaults to "positional_embeddings".
            dtype ([type], optional): [description]. Defaults to tf.float32.
        """
        super(ResNetBlock, self).__init__(name=name, dtype=dtype, **kwargs)

        self._out_ch = out_ch
        self._initializer = initializer
        self._bias_initializer = bias_initializer
        self._dtype = dtype
        self._use_dropout = use_dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        if activation is not None:
            self.activation = get_activation(activation)

        self.gp_norm1 = tfa.layers.GroupNormalization(name='group_norm_1')
        self.gp_norm2 = tfa.layers.GroupNormalization(name='group_norm_2')

        self.conv2d_layer1 = tf.keras.layers.Conv2D(
            out_ch, kernel_size=kernel_size, strides=strides, use_bias=True, padding='SAME', name='conv_1'
        )
        self.conv2d_layer2 = tf.keras.layers.Conv2D(
            out_ch, kernel_size=kernel_size, strides=strides, use_bias=True, padding='SAME', name='conv_2'
        )

        if self.use_scale_shift_norm:
            self.time_text_dense = tf.keras.layers.Dense(
                self._out_ch * 2,  # scale by 2
                use_bias=use_bias,
                activation=self.activation,
                kernel_initializer=self._initializer,
                bias_initializer=self._bias_initializer,
                name="time_text_dense",
            )
        else:
            self.time_text_dense = tf.keras.layers.Dense(
                self._out_ch,
                use_bias=use_bias,
                activation=self.activation,
                kernel_initializer=self._initializer,
                bias_initializer=self._bias_initializer,
                name="time_text_dense",
            )

        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self.conv1d = tf.keras.layers.Conv2D(
            self._out_ch, kernel_size=(1, 1), strides=(1, 1), use_bias=True, padding='SAME', name='conv_1d'
        )

    def build(self, input_shape):
        """Build variables based on shape at run time.

        Args:
            input_shape ([input_word_embeddings 3D, attention_mask 3D]): input_word_embeddings
            (b x s x h) and attention_mask (b x 1 x s)

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        image_shape, temb_shape = input_shape
        B, H, W, C = image_shape

        # If input channel doesn't match with output channel,
        # then we have to make use of extra dense layer to project the image,
        # so that we can sum it with hidden states
        self.dense_projection = tf.identity
        if C != self._out_ch:
            self.dense_projection = tf.keras.layers.Dense(
                self._out_ch,
                use_bias=True,
                activation=None,
                kernel_initializer=self._initializer,
                bias_initializer=self._bias_initializer,
                name="projection_dense",
            )

    def get_config(self):
        """Config based on init arguments

        Returns:
            [dict]: Dict of all init arguments
        """
        config = {
            "out_ch": self._out_ch,
            "initializer": self._initializer,
            "name": self._name,
            "dtype": self._dtype,
        }
        base_config = super(ResNetBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Call

        Args:
            inputs: List of 2 tensors [image, temb]
            image added with noise at a particular timestep - t. (b x h x w x num_channels)
            cemb : conditional embeddings (time_embeddings + text_pooled_embeddings )

        Returns:
            [tf.Tensor]: embeddings 3D (b x s x h)
        """
        image, cemb = inputs
        # Time and Text Projection
        cemb_projected = self.time_text_dense(cemb)

        h = self.activation(self.gp_norm1(image))
        h = self.conv2d_layer1(h)

        if self.use_scale_shift_norm:
            scale, shift = tf.split(cemb_projected, 2, axis=-1)
            scale = scale[:, None, None, :]  # Make it 4D (b x 1 x 1 x out_ch)
            shift = shift[:, None, None, :]  # Make it 4D (b x 1 x 1 x out_ch)
            h = self.activation(self.gp_norm2(h)) * (1 + scale) + shift
            h = self.dropout_layer(h, training=self._use_dropout)
            h = self.conv2d_layer2(h)
        else:
            # Add time embeddings and text pooled embeddings
            h += cemb_projected[:, None, None, :]  # Make it 4D (b x 1 x 1 x out_ch)

            h = self.activation(self.gp_norm2(h))
            h = self.dropout_layer(h, training=self._use_dropout)
            h = self.conv2d_layer2(h)

        # Skip Connection
        image = self.dense_projection(self.conv1d(image))
        out = image + h

        return out
