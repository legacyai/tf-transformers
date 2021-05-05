import tensorflow as tf
from absl import logging

from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.layers.mask import CrossAttentionMask

logging.set_verbosity("INFO")


class EncoderDecoder(LegacyLayer):
    def __init__(
        self,
        encoder,
        decoder,
        name=None,
        is_training=False,
        use_dropout=False,
        encoder_sequence_length=None,
        **kwargs,
    ):

        self.encoder = encoder
        self.decoder = decoder
        self.is_training = is_training
        self.use_dropout = use_dropout
        self.model_name = name
        self.encoder_sequence_length = encoder_sequence_length

        if not name.startswith("tf_transformers"):
            kwargs["name"] = "tf_transformers/" + self.model_name
        else:
            kwargs["name"] = self.model_name

        super(EncoderDecoder, self).__init__(is_training=self.is_training, use_dropout=self.use_dropout, **kwargs)

        # Two different hidden dimension has to be changed
        if self.encoder.embedding_size != self.decoder.embedding_size:
            self.encoder_decoder_projection = tf.keras.layers.Dense(self.decoder.embedding_size, activation="tanh")
        else:
            self.encoder_decoder_projection = tf.identity

        # Initialize model
        self.model_inputs, self.model_ouputs = self.get_model(
            encoder_sequence_length=self.encoder_sequence_length, initialize_only=True
        )
        logging.info("Initialized Variables")

    def get_model(self, encoder_sequence_length=None, initialize_only=False):
        """Overriding LegacyLayer
        embedding_sequence_length: For TFlite we need this value to be an int,d c not None
        """
        if self.is_training:
            encoder_inputs = self.encoder.model_inputs
            decoder_inputs = self.decoder.model_inputs

            main_inputs = {}
            for k, v in encoder_inputs.items():
                main_inputs["encoder_" + k] = v

            # We need to override this for tflite
            decoder_encoder_mask = decoder_inputs["decoder_encoder_mask"]
            batch, sequence_length, _ = decoder_encoder_mask.shape
            decoder_encoder_mask.set_shape((batch, sequence_length, encoder_sequence_length))
            decoder_inputs["decoder_encoder_mask"] = decoder_encoder_mask

            for k, v in decoder_inputs.items():
                if k in [
                    "encoder_hidden_states",
                    "decoder_encoder_mask",
                    "decoder_embeddings",
                ]:
                    # main_inputs[k] = v
                    continue
                main_inputs["decoder_" + k] = v

            layer_outputs = self(main_inputs)
            logging.info("Inputs -->")

            for k, v in main_inputs.items():
                logging.info("{} ---> {}".format(k, v))

            if initialize_only:
                return main_inputs, layer_outputs
        else:
            encoder_inputs = self.encoder.model_inputs
            decoder_inputs = self.decoder.model_inputs

            main_inputs = {}
            for k, v in encoder_inputs.items():
                main_inputs["encoder_" + k] = v

            for k, v in decoder_inputs.items():
                if k in [
                    "encoder_hidden_states",
                    "decoder_encoder_mask",
                    "decoder_embeddings",
                ]:
                    # main_inputs[k] = v
                    continue
                main_inputs["decoder_" + k] = v

            main_inputs["encoder_hidden_states"] = decoder_inputs["encoder_hidden_states"]
            layer_outputs = self(main_inputs)
            logging.info("Inputs -->")

            for k, v in main_inputs.items():
                logging.info("{} ---> {}".format(k, v))

        if initialize_only:
            # We just want to initialize variables
            return main_inputs, layer_outputs
        model = LegacyModel(inputs=main_inputs, outputs=layer_outputs, name=self.name)

        # Add config

        config = {}
        config["encoder"] = self.encoder._config_dict
        config["decoder"] = self.decoder._config_dict
        model.model_config = config
        return model

    def call_training(self, inputs):

        encoder_inputs = {
            k.replace("encoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("encoder_")
            if k not in ["encoder_hidden_states"]
        }
        decoder_inputs = {
            k.replace("decoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("decoder_")
            if k not in ["decoder_embeddings", "decoder_encoder_mask"]
        }

        decoder_encoder_mask = CrossAttentionMask()([decoder_inputs["input_ids"], encoder_inputs["input_mask"]])

        # Call Encoder and take the last hidden states (B x S x E)
        encoder_outputs = self.encoder(encoder_inputs)
        encoder_hidden_states = encoder_outputs["token_embeddings"]
        encoder_hidden_states = self.encoder_decoder_projection(encoder_hidden_states)

        decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_inputs["decoder_encoder_mask"] = decoder_encoder_mask

        decoder_outputs = self.decoder(decoder_inputs)
        decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
        return decoder_outputs

    def call_predict(self, inputs):

        encoder_inputs = {
            k.replace("encoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("encoder_")
            if k not in ["encoder_hidden_states"]
        }
        # encoder_inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"]

        decoder_inputs = {
            k.replace("decoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("decoder_")
            if k not in ["decoder_embeddings", "decoder_encoder_mask"]
        }
        decoder_encoder_mask = CrossAttentionMask()([decoder_inputs["input_ids"], encoder_inputs["input_mask"]])

        # While decoding we have to calculate it only once
        def use_cache_encoder():
            return tf.identity(inputs["encoder_hidden_states"])

        # First step of decoding process
        def calculate_encoder_hidden_state():
            encoder_outputs = self.encoder(encoder_inputs)
            encoder_hidden_states = self.encoder_decoder_projection(encoder_outputs["token_embeddings"])
            return tf.cast(encoder_hidden_states, tf.float32)

        cache_key = decoder_inputs["all_cache_key"]
        encoder_hidden_states = tf.cond(
            tf.equal(tf.reduce_sum(cache_key), 0.0),
            lambda: calculate_encoder_hidden_state(),
            lambda: use_cache_encoder(),
        )

        decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_inputs["decoder_encoder_mask"] = decoder_encoder_mask

        decoder_outputs = self.decoder(decoder_inputs)
        decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_outputs["decoder_all_cache_key"] = decoder_outputs["all_cache_key"]
        decoder_outputs["decoder_all_cache_value"] = decoder_outputs["all_cache_value"]

        del decoder_outputs["all_cache_key"]
        del decoder_outputs["all_cache_value"]

        return decoder_outputs

    def call(self, inputs):

        if self.is_training:
            outputs = self.call_training(inputs)
        else:
            outputs = self.call_predict(inputs)
        return outputs
