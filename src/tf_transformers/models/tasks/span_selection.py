import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel


class Span_Selection_Model(LegacyLayer):
    def __init__(self, model, use_all_layers=False, activation=None, **kwargs):
        super(Span_Selection_Model, self).__init__(**kwargs)
        self.model = model
        if isinstance(model, LegacyModel):
            self.model_config = model.model_config
        elif isinstance(model, tf.keras.layers.Layer):
            self.model_config = model._config_dict
        self.use_all_layers = use_all_layers
        self.logits_layer = tf.keras.layers.Dense(
            2,
            activation=activation,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )

        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def call(self, inputs):
        result = self.model(inputs)
        start_logits_outputs = []
        end_logits_outputs = []
        if self.use_all_layers:
            # each layer token embeddings
            for token_embeddings in result["all_layer_token_embeddings"]:
                outputs = self.logits_layer(token_embeddings)
                start_logits = outputs[:, :, 0]
                end_logits = outputs[:, :, 1]
                start_logits_outputs.append(start_logits)
                end_logits_outputs.append(end_logits)
            return {"start_logits": start_logits_outputs, "end_logits": end_logits_outputs}

        else:
            # last layer token embeddings
            token_embeddings = result["token_embeddings"]
            outputs = self.logits_layer(token_embeddings)
            start_logits = outputs[:, :, 0]
            end_logits = outputs[:, :, 1]
            return {
                "start_logits": start_logits,
                "end_logits": end_logits,
            }

    def get_model(self, initialize_only=False):
        inputs = self.model.input
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name="span_selection")
        model.model_config = self.model_config
        return model
