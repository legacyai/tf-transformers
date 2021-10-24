import tensorflow as tf
from absl import logging

from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.layers.mask import CrossAttentionMask


def assert_shapes(encoder_embeddings, decoder_embeddings):
    """Assert shapes


    Args:
        encoder_embeddings tf.tensor
        decoder_embeddings tf.tensor

    Raises:
        ValueError: [description]
    """
    encoder_shape = list(encoder_embeddings.shape)
    decoder_shape = list(decoder_embeddings.shape)

    if encoder_shape != decoder_shape:
        raise ValueError("Encoder shape {} not matching with Decoder shape {}".format(encoder_shape, decoder_shape))
    else:
        True


def share_embedding_layers(encoder_layer, decoder_layer):
    """Share Embeddings Layers
    Args:
        encoder_layer ([tf.keras.layers.Layer]): Encoder
        decoder_layer ([tf.keras.layers.Layer]): Decoder
    """
    assert_shapes(encoder_layer._embedding_layer.embeddings, decoder_layer._embedding_layer.embeddings)
    decoder_layer._embedding_layer = encoder_layer._embedding_layer

    try:
        if encoder_layer._type_embeddings_layer and decoder_layer._type_embeddings_layer:
            assert_shapes(
                encoder_layer._type_embeddings_layer.embeddings, decoder_layer._type_embeddings_layer.embeddings
            )
            decoder_layer._type_embeddings_layer = encoder_layer._type_embeddings_layer
    except Exception as e:
        logging.info(e)

    try:
        if encoder_layer._positional_embedding_layer and decoder_layer._positional_embedding_layer:
            assert_shapes(
                encoder_layer._positional_embedding_layer.embeddings,
                decoder_layer._positional_embedding_layer.embeddings,
            )
            decoder_layer._positional_embedding_layer = encoder_layer._positional_embedding_layer
    except Exception as e:
        logging.info(e)


def share_encoder_layers(encoder_layer, decoder_layer):
    """Share Encoder layers with Decoder
    Args:
        encoder_layer ([tf.keras.layers.Layer]): Encoder
        decoder_layer ([tf.keras.layers.Layer]): Decoder
    """
    for i, enc_layer in enumerate(encoder_layer._transformer_layers):
        dec_layer = decoder_layer._transformer_layers[i]

        # If this doesn't match, we will be in trouble
        # if enc_layer._attention_layer._use_auto_regressive == dec_layer._attention_layer._use_auto_regressive:
        dec_layer._attention_layer = enc_layer._attention_layer

        dec_layer._attention_output_dense = enc_layer._attention_output_dense

        try:
            dec_layer._attention_layer_norm = enc_layer._attention_layer_norm
        except Exception as e:
            logging.warn(e)
        try:
            dec_layer._intermediate_dense = enc_layer._intermediate_dense
        except Exception as e:
            logging.warn(e)
        try:
            dec_layer._output_dense = enc_layer._output_dense
        except Exception as e:
            logging.warn(e)
        try:
            dec_layer._output_layer_norm = enc_layer._output_layer_norm
        except Exception as e:
            logging.warn(e)

    try:
        if encoder_layer._pooler_layer and decoder_layer._pooler_layer:
            decoder_layer._pooler_layer = encoder_layer._pooler_layer
    except Exception as e:
        logging.warn(e)

    try:
        if encoder_layer._use_mlm_layer and decoder_layer._use_mlm_layer:
            if encoder_layer._masked_lm_layer and decoder_layer._masked_lm_layer:
                decoder_layer._masked_lm_layer = encoder_layer._masked_lm_layer
            if encoder_layer._masked_lm_bias and decoder_layer._masked_lm_bias:
                decoder_layer._masked_lm_bias = encoder_layer._masked_lm_bias
    except Exception as e:
        logging.warn(e)


class EncoderDecoder(LegacyLayer):
    """Encoder Decoder Model"""

    def __init__(
        self,
        encoder,
        decoder,
        share_embeddings=False,
        share_encoder=False,
        is_training=False,
        use_dropout=False,
        encoder_sequence_length=None,
        decoder_start_token_id=None,
        **kwargs,
    ):
        """
        Args:
            encoder ([tf.keras.layers.Layer]): [Encoder]
            decoder ([tf.keras.layers.Layer]): [Decoder]
            share_embeddings (bool, optional): [To share embeddings from encoder to decoder]. Defaults to False.
            share_encoder (bool, optional): [To share most layer from encoder to decoder.
            But not cross attention layers]  Defaults to False.
            is_training (bool, optional): [description]. Defaults to False.
            use_dropout (bool, optional): [description]. Defaults to False.
            encoder_sequence_length ([type], optional): [Max length of encoder]. Defaults to None.
        """
        self._encoder = encoder
        self._decoder = decoder
        self._share_embeddings = share_embeddings
        self._share_encoder = share_encoder
        self._is_training = is_training
        self._use_dropout = use_dropout
        self._encoder_sequence_length = encoder_sequence_length
        self.decoder_start_token_id = decoder_start_token_id

        self._encoder_config_dict = self._encoder._config_dict
        self._decoder_config_dict = self._decoder._config_dict

        self._model_name = self._encoder.name + "_" + self._decoder.name
        self._model_name = self._model_name.replace("tf_transformers", "").replace("/", "")

        if self._share_embeddings:
            share_embedding_layers(self._encoder, self._decoder)

        if self._share_encoder:
            share_encoder_layers(self._encoder, self._decoder)

        super(EncoderDecoder, self).__init__(
            is_training=self._is_training, use_dropout=self._use_dropout, name=self._model_name, **kwargs
        )

        if self._encoder._mask_mode != "user_defined":
            raise ValueError("mask_mode for encoder should be `user-defined`.")
        # Two different hidden dimension has to be changed
        if self._encoder_config_dict["embedding_size"] != self._decoder_config_dict["embedding_size"]:
            self._encoder_decoder_projection = tf.keras.layers.Dense(
                self._encoder_config_dict["embedding_size"], activation="linear"
            )
        else:
            self._encoder_decoder_projection = tf.identity

        # Initialize model
        self.model_inputs, self.model_ouputs = self.get_model(initialize_only=True)

    def get_model(self, initialize_only=False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: model (tf.keras.Layer) instance
            initialize_only: bool
        """
        # encoder_sequence_length = self._encoder._sequence_length
        # decoder_sequence_length = self._decoder._sequence_length

        encoder_inputs = self._encoder.model_inputs
        decoder_inputs = self._decoder.model_inputs

        inputs = {}

        # Convert 'input_ids' --> 'encoder_input_ids'
        # Add 'encoder' prefix
        for k, v in encoder_inputs.items():
            if k in ["input_ids", "input_mask", "input_type_ids"]:
                shape = encoder_inputs[k].shape
                inputs["encoder_" + k] = tf.keras.layers.Input(
                    shape[1:], batch_size=encoder_inputs[k].shape[0], name="encoder_" + k, dtype=encoder_inputs[k].dtype
                )
                continue
            inputs["encoder_" + k] = v

        # Convert 'input_ids' --> 'decoder_input_ids'
        # Add 'decoder' prefix
        for k, v in decoder_inputs.items():
            # Do not add prefix if 'decoder' or 'encoder' is present
            if k.startswith("decoder") or k.startswith("encoder"):
                inputs[k] = v
                continue
            # We don't want to change name of all_cache_key to decoder_all_cache_key
            # and similarily for all_cache_value . As it will raise issues
            # while serializing
            if k in ["all_cache_key", "all_cache_value"]:
                shape = decoder_inputs[k].shape
                inputs["decoder_" + k] = tf.keras.layers.Input(
                    shape[1:], batch_size=decoder_inputs[k].shape[0], name="decoder_" + k, dtype=decoder_inputs[k].dtype
                )
                continue
            inputs["decoder_" + k] = v

        del inputs["decoder_encoder_mask"]
        if not self._decoder._use_auto_regressive:
            del inputs["encoder_hidden_states"]
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        # Add config
        config = {}
        config["encoder"] = self._encoder._config_dict
        # Check for decoder start token id in decoder
        if "decoder_start_token_id" not in self._decoder._config_dict:
            if self.decoder_start_token_id is None:
                raise ValueError(
                    "In EncoderDecoder setting, `decoder_start_token_id` has to set either from config or\
                    constructor. Assuming we are in Auto Regressive setting"
                )
            else:
                logging.info("Setting decoder_start_token_id = {}".format(self.decoder_start_token_id))
                self._decoder._config_dict["decoder_start_token_id"] = self.decoder_start_token_id
        else:
            # If it is None in config
            if self._decoder._config_dict["decoder_start_token_id"] is None:
                if self.decoder_start_token_id is None:
                    raise ValueError(
                        "In EncoderDecoder setting, `decoder_start_token_id` has to set either from config or\
                        constructor. Assuming we are in Auto Regressive setting"
                    )
                else:
                    # If it is set in encoder-decoder constructor use it
                    logging.info("Setting decoder_start_token_id = {}".format(self.decoder_start_token_id))
                    self._decoder._config_dict["decoder_start_token_id"] = self.decoder_start_token_id

        config["decoder"] = self._decoder._config_dict
        model.model_config = config
        return model

    def call_forward(self, inputs):
        """Forward pass of an EncoderDecoder Model

        Args:
            inputs ([dict of tf.Tensor]): This is the input to the model.

            'encoder_input_ids'         --> tf.int32 (b x s)
            'encoder_input_mask'        --> tf.int32 (b x s) # optional
            'encoder_input_type_ids'    --> tf.int32 (b x s) # optional

            'decoder_input_ids'         --> tf.int32 (b x s)
            'decoder_input_mask'        --> tf.int32 (b x s) # optional
            'decoder_input_type_ids'    --> tf.int32 (b x s) # optional

        Returns:
            [dict of tf.Tensor]: Output from the model

            'cls_output'        --> tf.float32 (b x s) # optional
            'token_embeddings'  --> tf.float32 (b x s x h)
            'all_layer_token_embeddings' --> tf.float32 (List of (b x s x h)
                                              from all layers)
            'all_layer_cls_output'       --> tf.float32 (List of (b x s)
                                              from all layers)
        """
        # Replace 'encoder_input_ids'      to 'input_ids'
        # Replace 'encoder_input_mask'     to 'input_mask'
        # Replace 'encoder_input_type_ids' to 'input_type_ids'
        encoder_inputs = {
            k.replace("encoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("encoder_")
            if k not in ["encoder_hidden_states"]
        }
        # Replace 'decoder_input_ids'      to 'input_ids'
        # Replace 'decoder_input_mask'     to 'input_mask'
        # Replace 'decoder_input_type_ids' to 'input_type_ids'
        decoder_inputs = {
            k.replace("decoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("decoder_")
            if k not in ["decoder_encoder_mask"]
        }

        # Call Encoder and take the last hidden states (B x S x E)
        encoder_outputs = self._encoder(encoder_inputs)
        encoder_hidden_states = encoder_outputs["token_embeddings"]
        encoder_hidden_states = self._encoder_decoder_projection(encoder_hidden_states)

        # This is decoder_encoder_mask
        decoder_encoder_mask = CrossAttentionMask()([decoder_inputs["input_ids"], encoder_inputs["input_mask"]])

        # Add the inputs to decoder
        decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_inputs["decoder_encoder_mask"] = decoder_encoder_mask

        decoder_outputs = self._decoder(decoder_inputs)
        decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
        return decoder_outputs

    def call_auto_regressive(self, inputs):
        """Forward pass of an EncoderDecoder Model

        Args:
            inputs ([dict of tf.Tensor]): This is the input to the model.

            'encoder_input_ids'         --> tf.int32 (b x s)
            'encoder_input_mask'        --> tf.int32 (b x s) # optional
            'encoder_input_type_ids'    --> tf.int32 (b x s) # optional

            'decoder_input_ids'         --> tf.int32 (b x s)
            'decoder_input_mask'        --> tf.int32 (b x s) # optional
            'decoder_input_type_ids'    --> tf.int32 (b x s) # optional

            'decoder_all_cache_key'     --> tf.float32 (num_hidden_layers ,
                                     batch_size ,
                                     num_attention_heads ,
                                     sequence_length,
                                     attention_head_size)

            'decoder_all_cache_value'    --> tf.float32 (num_hidden_layers ,
                                     batch_size ,
                                     num_attention_heads ,
                                     sequence_length,
                                     attention_head_size)
        Returns:
            [dict of tf.Tensor]: Output from the model

            'cls_output'        --> tf.float32 (b x s) # optional
            'token_embeddings'  --> tf.float32 (b x s x h)
            'all_layer_token_embeddings' --> tf.float32 (List of (b x s x h)
                                              from all layers)
            'all_layer_cls_output'       --> tf.float32 (List of (b x s)
                                              from all layers)
        """
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
            if k not in ["decoder_encoder_mask"]
        }
        decoder_encoder_mask = CrossAttentionMask()([decoder_inputs["input_ids"], encoder_inputs["input_mask"]])

        # While decoding we have to calculate it only once
        def use_cache_encoder():
            return tf.identity(inputs["encoder_hidden_states"])

        # First step of decoding process
        def calculate_encoder_hidden_state():
            encoder_outputs = self._encoder(encoder_inputs)
            encoder_hidden_states = self._encoder_decoder_projection(encoder_outputs["token_embeddings"])
            return tf.cast(encoder_hidden_states, tf.float32)

        cache_key = decoder_inputs["all_cache_key"]
        encoder_hidden_states = tf.cond(
            tf.equal(tf.reduce_sum(cache_key), 0.0),
            lambda: calculate_encoder_hidden_state(),
            lambda: use_cache_encoder(),
        )

        decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_inputs["decoder_encoder_mask"] = decoder_encoder_mask

        decoder_outputs = self._decoder(decoder_inputs)
        decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_outputs["decoder_all_cache_key"] = decoder_outputs["all_cache_key"]
        decoder_outputs["decoder_all_cache_value"] = decoder_outputs["all_cache_value"]

        del decoder_outputs["all_cache_key"]
        del decoder_outputs["all_cache_value"]

        return decoder_outputs

    def call(self, inputs):
        """Call

        Args:
            inputs : dict of tf.tensor

        Returns:
            dict of tf.tensor
        """
        if self._is_training is False and self._decoder._use_auto_regressive:
            outputs = self.call_auto_regressive(inputs)
        else:
            outputs = self.call_forward(inputs)
        return outputs
