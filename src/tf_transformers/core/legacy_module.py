import tensorflow as tf


class LegacyModuleCustom(tf.Module):
    def __init__(self, model, name=None):
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

    @tf.function
    def __call__(self, **kwargs):
        return self.model(kwargs)

    def save(self, save_dir, signature_name="serving_default"):
        call_output = self.__call__.get_concrete_function(**self.model.input)
        tf.saved_model.save(self, save_dir, signatures={signature_name: call_output})


class LegacyModule(tf.Module):
    def __init__(self, model, name=None):
        super(LegacyModule, self).__init__(name=name)
        self.model = model

    @tf.function
    def __call__(self, **kwargs):
        return self.model(kwargs)

    def save(self, save_dir, signature_name="serving_default"):
        call_output = self.__call__.get_concrete_function(**self.model.input)
        tf.saved_model.save(self, save_dir, signatures={signature_name: call_output})


# Old Way
# class LegacyModule(tf.Module):
#     def __init__(self, model, name=None):
#         super(LegacyModule, self).__init__(name=name)
#         self.model = model

#     @tf.function
#     def __call__(self, **kwargs):
#         return self.model(kwargs)

# gpt2_module = LegacyModule(model_tf_transformers)
# call_output = gpt2_module.__call__.get_concrete_function(**model_tf_transformers.input)
# module_output_path = "model_pb"
# tf.saved_model.save(
#     gpt2_module, module_output_path, signatures={"serving_default": call_output}
# )
