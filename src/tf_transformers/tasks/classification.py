import tensorflow as tf
from tf_transformers.core import LegacyModel, LegacyLayer


class Classification_Model(LegacyLayer):
    def __init__(self, model, num_classes, use_all_layers=False, activation=None, **kwargs):
        super(Classification_Model, self).__init__(**kwargs)
        self.model = model
        if isinstance(model, LegacyModel):
            self.model_config = model.model_config
        elif isinstance(model, tf.keras.layers.Layer):
            self.model_config = model._config_dict
        self.use_all_layers = use_all_layers
        self.logits_layer = tf.keras.layers.Dense(
            num_classes,
            activation=activation,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )

    def call(self, inputs):
        result = self.model(inputs)
        logits_outputs = []
        if self.use_all_layers:
            # each layer token embeddings
            for token_embeddings in result["all_layer_cls_output"]:
                outputs = self.logits_layer(token_embeddings)
                logits_outputs.append(outputs)
            return {"class_logits": logits_outputs}

        else:
            # last layer token embeddings
            token_embeddings = result["cls_output"]
            outputs = {"class_logits": self.logits_layer(token_embeddings)}
            return outputs

    def get_model(self):
        layer_output = self(self.model.input)
        model = LegacyModel(inputs=self.model.input, outputs=layer_output, name="span_selection")
        model.model_config = self.model_config
        return model
