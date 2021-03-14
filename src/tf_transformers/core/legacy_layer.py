import tensorflow as tf
from absl import logging

from tf_transformers.core.legacy_model import LegacyModel
from tf_transformers.layers import MLMLayer, OnDeviceEmbedding, SimplePositionEmbedding


class LegacyLayer(tf.keras.layers.Layer):
    """
    This layer is an extension of tf.keras.layers.Layer , by adding extra attributes.
    """

    def __init__(self, is_training=None, use_dropout=True, **kwargs):
        self.is_training = is_training
        self.use_dropout = use_dropout
        if self.is_training:
            pass
        else:
            self.use_dropout = False
        if self.is_training is None:
            raise AttributeError("`is_training` should be set to either [True or False]")
        super(LegacyLayer, self).__init__(**kwargs)

    def add_special_tokens(self):
        """
        Adding special tokens will be changing the word embeddings shape
        """

        raise NotImplementedError

    def validate_and_set_inputs(self):

        if self.mask_mode not in ["user_defined", "causal", "prefix"]:
            raise ValueError(
                "Unknown mask_mode `{}`provided. Supported modes are `{}`".format(
                    self.mask_mode, ["user_defined", "causal", "prefix"]
                )
            )
        if self.model_name is None:
            raise ValueError("`name` cannot be None. Please provide a meaningful name")
        if self.is_training is None:
            raise ValueError("`is_training` cannot be None. Please provide a `True` or `False`")
        if self.mask_mode is None:
            raise ValueError("`mask_mode` cannot be None. Please provide `['user_defined', 'causal', 'prefix']`")

        # If `is_training` is False and `pipeline is None` means, we are using it for inference.
        # We will forcefully set it back to `is_training` is True and `use_dropout` is False.
        # For encoder-decoder models this is the encoder mode params.
        # Same mode will also be applicable for classification, QA etc.

        # pipeline_mode has only one value (now)
        if self.pipeline_mode and self.pipeline_mode != "auto-regressive":
            raise ValueError("`pipeline-mode` accepts either None or `auto-regressive` as values")

        # pipeline_mode is only for self.is_training = False
        if self.is_training:
            if self.pipeline_mode is not None:
                raise ValueError(
                    "When `is_training` is True, `pipeline_mode` \
                        should be None. But rather got `pipeline_mode` as {}".format(
                        self.pipeline_mode
                    )
                )
        # If both of these are None / False, which means we are using it mostly
        # for non auto-regressive tasks
        if self.is_training is False:
            if self.pipeline_mode is None:
                logging.info(
                    "We are overwriding `is_training` is False to \
                        `is_training` to True with `use_dropout` is False, no effects on your inference pipeline"
                )
                self.is_training = True
                self.use_dropout = False

        # Decoder Mode
        if self.is_decoder:
            # Only one should be True (not both)
            if self.cross_attention_inside_encoder:
                raise ValueError(
                    "When `is_decoder` is True `cross_attention_inside_encoder`\
                     should be False and vice versa "
                )
            # Decoder will never have a prefix model for time being
            # Needs to investigate
            if self.mask_mode == "prefix":
                raise ValueError(
                    "As you are in Decoder Mode (`is_decoder` is True), \
                    {} mask_mode doesn't make sense. \
                    For Decode `mask_mode` should be `causal` or `user_defined` ".format(
                        self.mask_mode
                    )
                )
            # If predict pipeline (auto regressive models need causal mask)
            if self.is_training is False:
                # Auto Regressive setting should only support causal mode
                if self.pipeline_mode == "auto-regressive":
                    if self.mask_mode != "causal":
                        raise ValueError(
                            "As you are in Decoder Mode  and \
                            auto-regressive pipeline(`is_decoder` is True), {} \
                            mask_mode doesn't make sense. \
                            For Decode `mask_mode` should be `causal` ".format(
                                self.mask_mode
                            )
                        )

    def get_embedding_layers(self, embedding_size=None):
        if embedding_size is None:
            embedding_size = self.embedding_size
        # Word Embedding Layer
        self._embedding_layer = OnDeviceEmbedding(
            vocab_size=self.vocab_size,
            embedding_width=embedding_size,
            initializer=self.initializer,
            name="word_embeddings",
        )
        self._type_embeddings = None
        self._position_embedding_layer = None
        if self.use_type_embeddings:
            # Type Embeddings
            self._type_embeddings = OnDeviceEmbedding(
                vocab_size=self.type_vocab_size,
                embedding_width=embedding_size,
                initializer=self.initializer,
                name="type_embeddings",
            )
        if self.use_positonal_embeddings:
            # Positional Embedding
            self._position_embedding_layer = SimplePositionEmbedding(
                initializer=self.initializer,
                max_sequence_length=self.max_position_embeddings,
                embedding_width=embedding_size,
                name="positional_embeddings",
            )
        return self._embedding_layer, self._type_embeddings, self._position_embedding_layer

    def get_call_method(self):

        # Training Pipeline
        if self.is_training:
            # Decoder Mode Training
            if self.is_decoder:
                method = self.call_decoder
            # If cross attention inside encoder
            elif self.cross_attention_inside_encoder:
                # Check if cross attention in encoder is True
                # irrespective of self.is_training we go here
                method = self.call_cross_attention_encoder

            # Encoder Mode / Mode for anything except Decoder / Cross attention
            else:
                # Note: This mode is also be used with `use_dropout`
                # is False, when `is_training` is False and `pipeline_mode` is None
                # Default (Training/ Predict) pipeline
                method = self.call_training

        # Predict Pipeline ( For decoder and auto-regressive mode only)
        else:
            # Decoder Predict pipeline
            if self.is_decoder:
                if self.pipeline_mode == "auto-regressive":
                    method = self.call_decoder_predict
                else:
                    # is_training with use_dropout = False
                    method = self.call_decoder
            elif self.cross_attention_inside_encoder:
                # we want predict mode to be ready only when
                # pipeline_mode == 'auto-regressive
                if self.pipeline_mode == "auto-regressive":
                    method = self.call_cross_attention_encoder_predict
                else:
                    self.use_dropout = False
                    method = self.call_cross_attention_encoder
            else:
                # Encoder Model (Predict pipeline) All models can be in this form
                if self.pipeline_mode == "auto-regressive":
                    method = self.call_predict

        return method

    def get_model(self, initialize_only=False):
        """Convert tf.keras.Layer to a tf.keras.Model object.
        Args:
            self: model (tf.keras.Layer) instance
        """
        inputs = {}

        if self.cross_attention_inside_encoder:
            inputs["encoder_input_ids"] = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="encoder_input_ids",
            )
            inputs["decoder_input_ids"] = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="decoder_input_ids",
            )
            if self.use_type_embeddings:
                inputs["encoder_input_type_ids"] = tf.keras.layers.Input(
                    shape=(self.sequence_length,),
                    batch_size=self.batch_size,
                    dtype=tf.int32,
                    name="encoder_input_type_ids",
                )
                inputs["decoder_input_type_ids"] = tf.keras.layers.Input(
                    shape=(self.sequence_length,),
                    batch_size=self.batch_size,
                    dtype=tf.int32,
                    name="decoder_input_type_ids",
                )
            if self.mask_mode in ["user_defined", "prefix"]:
                inputs["encoder_input_mask"] = tf.keras.layers.Input(
                    shape=(self.sequence_length,),
                    batch_size=self.batch_size,
                    dtype=tf.int32,
                    name="encoder_input_mask",
                )
            if self.is_training is False:
                inputs["decoder_all_cache_key"] = tf.keras.layers.Input(
                    shape=(
                        None,
                        self.num_attention_heads,
                        None,
                        self.self.attention_head_size,
                    ),
                    dtype=tf.float32,
                    name="decoder_all_cache_key",
                )
                inputs["decoder_all_cache_value"] = tf.keras.layers.Input(
                    shape=(
                        None,
                        self.num_attention_heads,
                        None,
                        self.self.attention_head_size,
                    ),
                    dtype=tf.float32,
                    name="decoder_all_cache_value",
                )
                # self.num_hidden_layers x batch_size x sequence_length x embedding_size
                inputs["encoder_hidden_states"] = tf.keras.layers.Input(
                    shape=(self.sequence_length, self.embedding_size),
                    batch_size=self.batch_size,
                    dtype=tf.float32,
                    name="encoder_hidden_states",
                )

            layer_outputs = self(inputs)
            # We just want to initialize variables
            if initialize_only:
                return inputs, layer_outputs
            # logging.info("Inputs -->")
            # for k, v in inputs.items():
            #     logging.info("{} ---> {}".format(k, v))
            model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self.name)
            model.model_config = {"decoder": self._config_dict}
            return model

        # Encoder
        if self.is_decoder is False:
            input_ids = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="input_ids",
            )
            input_mask = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="input_mask",
            )
            input_type_ids = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="input_type_ids",
            )

            inputs["input_ids"] = input_ids
            # When `mask_mode` is `causal` , input_mask is not required
            if self.mask_mode in ["user_defined", "prefix"]:
                inputs["input_mask"] = input_mask
            # Default True in BERT
            if self.use_type_embeddings:
                inputs["input_type_ids"] = input_type_ids

            if self.is_training is False:

                if self.pipeline_mode == "auto-regressive":
                    # Batch size is None
                    # (12 , None , 12 , None, 64)
                    # (self.num_hidden_layers,
                    # batch_size,
                    # self.num_attention_heads,
                    # sequence_length,
                    # self.embedding_size//self.num_attention_heads)
                    all_cache_key = tf.keras.layers.Input(
                        shape=(
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_head_size,
                        ),
                        dtype=tf.float32,
                        name="all_cache_key",
                    )
                    all_cache_value = tf.keras.layers.Input(
                        shape=(
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_head_size,
                        ),
                        dtype=tf.float32,
                        name="all_cache_value",
                    )
                    # Here batch_size = 1 , means we are dealing with vector for past_length
                    past_length = tf.keras.layers.Input(shape=(None,), batch_size=1, dtype=tf.int32, name="past_length")
                    inputs["all_cache_key"] = all_cache_key
                    inputs["all_cache_value"] = all_cache_value
                    inputs["past_length"] = past_length

        else:
            input_ids = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="decoder_input_ids",
            )
            input_mask = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="decoder_input_mask",
            )
            input_type_ids = tf.keras.layers.Input(
                shape=(self.sequence_length,),
                batch_size=self.batch_size,
                dtype=tf.int32,
                name="decoder_input_type_ids",
            )
            encoder_hidden_states = tf.keras.layers.Input(
                shape=(self.sequence_length, self.embedding_size),
                batch_size=self.batch_size,
                dtype=tf.float32,
                name="encoder_hidden_states",
            )
            # batch_size x decoder_input_length x encoder_input_length
            decoder_encoder_mask = tf.keras.layers.Input(
                shape=(self.sequence_length, None),
                batch_size=self.batch_size,
                dtype=tf.float32,
                name="decoder_encoder_mask",
            )

            inputs["input_ids"] = input_ids
            # When `mask_mode` is `causal` , input_mask is not required
            if self.mask_mode in ["user_defined", "prefix"]:
                inputs["input_mask"] = input_mask
            # Default True in BERT
            if self.use_type_embeddings:
                inputs["input_type_ids"] = input_type_ids

            inputs["encoder_hidden_states"] = encoder_hidden_states
            inputs["decoder_encoder_mask"] = decoder_encoder_mask

            if self.is_training is False:
                if self.pipeline_mode == "auto-regressive":
                    # Batch size is None
                    # (12 , None , 12 , None, 64)
                    # (self.num_hidden_layers,
                    # batch_size,
                    # self.num_attention_heads,
                    # sequence_length,
                    # self.embedding_size//self.num_attention_heads)
                    all_cache_key = tf.keras.layers.Input(
                        shape=(
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_head_size,
                        ),
                        dtype=tf.float32,
                        name="all_cache_key",
                    )
                    all_cache_value = tf.keras.layers.Input(
                        shape=(
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_head_size,
                        ),
                        dtype=tf.float32,
                        name="all_cache_value",
                    )
                    inputs["all_cache_key"] = all_cache_key
                    inputs["all_cache_value"] = all_cache_value

        layer_outputs = self(inputs)
        # We just want to initialize variables
        if initialize_only:
            return inputs, layer_outputs
        # logging.info("Inputs -->")
        # for k, v in inputs.items():
        #     logging.info("{} ---> {}".format(k, v))
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self.name)
        model.model_config = self._config_dict
        return model
