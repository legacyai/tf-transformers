import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel


class Classification_Model(LegacyLayer):
    def __init__(
        self,
        model,
        num_classes,
        activation=None,
        is_training=False,
        use_dropout=False,
        use_bias=True,
        kernel_initializer="truncated_normal",
        dropout_rate=0.2,
        classification_network=None,
        key='cls_output',
        **kwargs,
    ):
        r"""
        Simple Classification using Keras Layer

        Args:
            model (:obj:`LegacyLayer/LegacyModel`):
                Model.
                Eg:`~tf_transformers.model.BertModel`.
            num_classes (:obj:`int`):
                Number of classes
            activation (:obj:`str/tf.keras.Activation`, `optional`, defaults to None): Activation
            is_training (:obj:`bool`, `optional`, defaults to False): To train
            use_dropout (:obj:`bool`, `optional`, defaults to False): Use dropout
            use_bias (:obj:`bool`, `optional`, defaults to True): use bias
            dropout_rate (:obj: `float`, defaults to `0.2`)
            key (:obj: `str`, `optional`, defaults to 128): If specified, we use this
            key in model output dict and pass it through classfication layer. If its a list
            we return a list of logits for joint loss.
            kernel_initializer (:obj:`str/tf.keras.intitalizers`, `optional`, defaults to `truncated_normal`): Initializer for
            classification layer
            classfication_network (:obj: `tf.keras.Layer/Model`, `optional`, defaults to None): If a network is present,
            then we use this network for default classification layer. This network can
            be eg: tf.keras.Sequential([layer1, layer2]) etc. Make sure, output layer matches
            num_classes.
        """
        super(Classification_Model, self).__init__(
            is_training=is_training, use_dropout=use_dropout, name=model.name, **kwargs
        )

        self.model = model
        if isinstance(model, LegacyModel):
            self.model_config = model.model_config
        elif isinstance(model, tf.keras.layers.Layer):
            self.model_config = model._config_dict
        self._is_training = is_training
        self._use_dropout = use_dropout
        self._key = key

        if classification_network:
            self.classification_layer = classification_network
        else:
            self.classification_layer = tf.keras.layers.Dense(
                num_classes,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer="zeros",
            )
        self._classification_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def call(self, inputs):
        """Call"""
        model_outputs = self.model(inputs)

        if self._key not in model_outputs:
            raise ValueError("Specified key `{}` not found in model output dict".format(self._key))

        if self._key in model_outputs:
            # if list return list of class logits
            if isinstance(model_outputs[self._key], list):
                logits_outputs = []
                # each layer token embeddings
                for per_layer_output in model_outputs[self._key]:
                    outputs = self.classification_layer(per_layer_output)
                    outputs = self._classification_dropout(outputs, training=self._use_dropout)
                    logits_outputs.append(outputs)
                return {"class_logits": logits_outputs}

            else:
                # last layer token embeddings
                outputs = model_outputs[self._key]
                outputs = self.classification_layer(outputs)
                outputs = self._classification_dropout(outputs)
                outputs = {"class_logits": outputs}
                return outputs

    def get_model(self, initialize_only=False):
        """Get model"""
        inputs = self.model.input
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name="classification")
        model.model_config = self.model_config
        return model
