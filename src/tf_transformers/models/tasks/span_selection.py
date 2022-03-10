import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel


class Span_Selection_Model(LegacyLayer):
    def __init__(
        self,
        model,
        activation=None,
        is_training=False,
        use_dropout=False,
        use_bias=True,
        kernel_initializer="truncated_normal",
        dropout_rate=0.2,
        span_selection_network=None,
        key='token_embeddings',
        **kwargs,
    ):
        r"""
        Simple Span Selection model (QA) using Keras Layer

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
        super(Span_Selection_Model, self).__init__(
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

        if span_selection_network:
            self.span_selection_layer = span_selection_network
        else:
            self.span_selection_layer = tf.keras.layers.Dense(
                2,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer="zeros",
            )
        self._span_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

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
                # each layer token embeddings
                start_logits_outputs = []
                end_logits_outputs = []
                for per_layer_output in model_outputs[self._key]:

                    outputs = self._span_dropout(self.span_selection_layer(per_layer_output))
                    start_logits = outputs[:, :, 0]
                    end_logits = outputs[:, :, 1]
                    start_logits_outputs.append(start_logits)
                    end_logits_outputs.append(end_logits)
                return {"start_logits": start_logits_outputs, "end_logits": end_logits_outputs}
            else:
                # last layer token embeddings
                outputs = model_outputs[self._key]
                outputs = self.span_selection_layer(outputs)
                outputs = self._span_dropout(outputs)
                start_logits = outputs[:, :, 0]
                end_logits = outputs[:, :, 1]
                return {"start_logits": start_logits, "end_logits": end_logits}

    def get_model(self, initialize_only=False):
        """Get model"""
        inputs = self.model.input
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name="classification")
        model.model_config = self.model_config
        return model
