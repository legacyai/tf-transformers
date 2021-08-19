import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel


class Token_Classification_Model(LegacyLayer):
    def __init__(self, model, token_vocab_size, use_all_layers=False, activation=None, **kwargs):
        super(Token_Classification_Model, self).__init__(**kwargs)
        self.model = model
        if isinstance(model, LegacyModel):
            self.model_config = model.model_config
        elif isinstance(model, tf.keras.layers.Layer):
            self.model_config = model._config_dict
        self.use_all_layers = use_all_layers
        self.logits_layer = tf.keras.layers.Dense(token_vocab_size, activation=activation)

    def call(self, inputs):
        result = self.model(inputs)
        token_logits = []
        if self.use_all_layers:
            # each layer token embeddings
            for token_embeddings in result["all_layer_token_embeddings"]:
                outputs = self.logits_layer(token_embeddings)
                token_logits.append(outputs)
            return {"token_logits": token_logits}

        else:
            # last layer token embeddings
            token_embeddings = result["token_embeddings"]
            outputs = self.logits_layer(token_embeddings)
            return {"token_logits": outputs}

    def get_model(self):
        layer_output = self(self.model.input)
        model = LegacyModel(inputs=self.model.input, outputs=layer_output, name="token_classification")
        model.model_config = self.model_config
        return model
