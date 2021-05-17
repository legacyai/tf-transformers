import tensorflow as tf


class LegacyModuleCustom(tf.Module):
    """LegacyModuleCustom : This will serialize by making use of
    config dict from the models, so that we can infer inputs shapes
    from serialized models.

    """

    def __init__(self, model, name=None):

        """
        Args:
            model (tf.keras.Model): Model
            name (str), optional): Model name
        """
        super(LegacyModuleCustom, self).__init__(name=name)
        self.model = model
        self.config = {}
        # Possible an Encoder Decoder model
        if "decoder" in model.model_config:
            model_config = model.model_config["decoder"]
        else:
            model_config = model.model_config
        self.config["embedding_size"] = tf.Variable(model_config["embedding_size"], name="embedding_size")
        self.config["num_attention_heads"] = tf.Variable(
            model_config["num_attention_heads"], name="num_attention_heads"
        )
        self.config["num_hidden_layers"] = tf.Variable(model_config["num_hidden_layers"], name="num_hidden_layers")
        self.config["attention_head_size"] = tf.Variable(
            model_config["attention_head_size"], name="attention_head_size"
        )

        self.config["embedding_size"] = tf.constant(768, name="embedding_size2")
        self.embedding_size = tf.Variable(768, name="embedding_size2")

    @tf.function
    def __call__(self, **kwargs):
        return self.model(kwargs)

    def save(self, save_dir, signature_name="serving_default"):
        input_spec = {name: keras_input.type_spec for name, keras_input in self.model.input.items()}
        call_output = self.__call__.get_concrete_function(**input_spec)
        tf.saved_model.save(self, save_dir, signatures={signature_name: call_output})


class LegacyModule(tf.Module):
    """LegacyModule : To make output names compatible we use this Module."""

    def __init__(self, model, name=None):
        super(LegacyModule, self).__init__(name=name)
        self.model = model

    @tf.function
    def __call__(self, **kwargs):
        return self.model(kwargs)

    def save(self, save_dir, signature_name="serving_default"):
        input_spec = {name: keras_input.type_spec for name, keras_input in self.model.input.items()}
        call_output = self.__call__.get_concrete_function(**input_spec)
        tf.saved_model.save(self, save_dir, signatures={signature_name: call_output})
