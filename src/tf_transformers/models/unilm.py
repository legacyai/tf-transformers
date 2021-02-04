from __future__ import absolute_import, division, print_function

import tensorflow as tf
from absl import logging

from tf_transformers.activations import get_activation
from tf_transformers.core import LegacyLayer
from tf_transformers.layers import (MLMLayer, OnDeviceEmbedding,
                                    SimplePositionEmbedding)
from tf_transformers.layers.mask import (CausalMask, SelfAttentionMask,
                                         prefix_mask)
from tf_transformers.layers.transformer import TransformerBERT

logging.set_verbosity("INFO")


class UNILMEncoder(LegacyLayer):
    """UniLM based encoder / Decoder .
    Unified Language Model Pre-training for Natural Language Understanding and Generation

    Authors: Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon # noqa

    Implementation of UNILM in TF2.0

    Paper: https://arxiv.org/abs/1905.03197
    Official Code: https://github.com/microsoft/unilm

    """

    def __init__(
        self,
        config,
        mask_mode="user_defined",
        name=None,
        use_dropout=False,
        is_training=None,
        batch_size=None,
        sequence_length=None,
        use_type_embeddings=True,
        use_positonal_embeddings=True,
        pipeline_mode=None,
        is_decoder=False,
        initialize_embeddings=False,
        model_dir=None,
        **kwargs,
    ):
        """
        Args:
            config: dict
            mask_mode: str, `user_defined` BERT by default uses masking for PADDED or MLM. But can be overridden . # noqa
            name: str, Name of the model
            use_dropout: bool, It is strictly optional. Sometimes, while training you can set `use_dropout` to False. # noqa
                         If `is_training` is False, `use_dropout` will be automatically set to False. # noqa
            batch_size: int, `batch_size` can be None or any int
            sequence_length: int, `sequence_length` can be None or any int
            use_type_embeddings: bool, By default BERT has type_embeddings, GPT2 don't.
            use_positonal_embeddings: bool, T5 don't have postional embeddings
            bidirectional: use in relative postional embedding (we can infer it based on mask_mode)
            is_decoder: bool, if True it will become decoder mode (as in Seq2Seq)
            initialize_embeddings: bool, When is_decoder = True, most cases, it will re-use the embedding layer from Encoder. # noqa
                            So. if you still want to initialize , set this to True # noqa
        """
        # Because saved_model causes some serialization problems here
        # self.config              = config
        self.vocab_size = config["vocab_size"]
        self.type_vocab_size = config["type_vocab_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.intermediate_size = config["intermediate_size"]
        self.embedding_size = config["embedding_size"]
        self.initializer_range = config["initializer_range"]
        self.hidden_act = config["hidden_act"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
        self.intermediate_act = config["intermediate_act"]
        self.layer_norm_epsilon = config["layer_norm_epsilon"]

        # Get activation and initiliazers
        self.activation = get_activation(self.hidden_act)
        self.intermediate_activation = get_activation(self.intermediate_act)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.initializer = tf.keras.initializers.get(initializer)
        self.mask_mode = mask_mode
        # If we use self.name , its a conflict with keras property
        self.model_name = name
        self.pipeline_mode = pipeline_mode
        self.is_decoder = is_decoder
        self.model_dir = model_dir

        if self.mask_mode not in ["user_defined", "causal", "prefix"]:
            raise ValueError(
                "Unknown mask_mode `{}`provided. Supported modes are `{}`".format(
                    self.mask_mode, ["user_defined", "causal", "prefix"]
                )
            )
        if self.model_name is None:
            raise ValueError("`name` cannot be None. Please provide a meaningful name")
        if is_training is None:
            raise ValueError("`is_training` cannot be None. Please provide a `True` or `False`")
        if self.mask_mode is None:
            raise ValueError("`mask_mode` cannot be None. Please provide `['user_defined', 'causal', 'prefix']`")

        # self._self_setattr_tracking = False
        self.mask_mode = mask_mode
        self.use_dropout = use_dropout
        self.is_training = is_training
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.use_type_embeddings = use_type_embeddings
        self.use_positonal_embeddings = use_positonal_embeddings
        self.initialize_embeddings = initialize_embeddings

        # If `is_training` is False and `pipeline is None` means, we are using it for inference.
        # We will forcefully set it back to `is_training` is True and `use_dropout` is False.
        # For encoder-decoder models # noqa
        # this is the encoder mode params. Same mode will also be applicable for classification, QA etc. # noqa

        if self.is_training:
            if self.pipeline_mode is not None:
                raise ValueError(
                    "When `is_training` is True, `pipeline_mode` should be None. \
                     But rather got `pipeline_mode` as {}".format(
                        self.pipeline_mode
                    )
                )

        if self.is_training is False:
            if self.pipeline_mode is None:
                logging.info(
                    "We are overwriding `is_training` is False to `is_training` to True \
                     with `use_dropout` is False, no effects on your inference pipeline"
                )
                self.is_training = True
                self.use_dropout = False

        # Decoder Mode
        if self.is_decoder:
            # Decoder will never have a prefix model for time being
            if self.mask_mode == "prefix":
                raise ValueError(
                    "As you are in Decoder Mode (`is_decoder` is True), {} mask_mode \
                     doesn't make sense. For Decode `mask_mode` \
                     should be `causal` or `user_defined` ".format(
                        self.mask_mode
                    )
                )
            # If predict pipeline
            if self.is_training is False:
                # Auto Regressive setting should only support causal mode
                if self.pipeline_mode == "auto-regressive":
                    if self.mask_mode != "causal":
                        raise ValueError(
                            "As you are in Decoder Mode  and auto-regressive \
                              pipeline(`is_decoder` is True), \
                              {} mask_mode doesn't make sense. For Decode \
                              `mask_mode` should be `causal` ".format(
                                self.mask_mode
                            )  # noqa
                        )

        if not name.startswith("tf_transformers"):
            kwargs["name"] = "tf_transformers/" + self.model_name
        else:
            kwargs["name"] = self.model_name
        super(UNILMEncoder, self).__init__(is_training=self.is_training, use_dropout=self.use_dropout, **kwargs)
        self._config_dict = {
            "initializer": tf.keras.initializers.serialize(initializer),
            "is_training": self.is_training,
            "use_dropout": self.use_dropout,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "name": kwargs["name"],
            "use_type_embeddings": self.use_type_embeddings,
            "use_positonal_embeddings": self.use_positonal_embeddings,
            "is_decoder": self.is_decoder,
            "initialize_embeddings": self.initialize_embeddings,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        }

        # Update config dict with passed config
        self._config_dict.update(config)

        if self.is_decoder:
            if self.initialize_embeddings:
                # Word Embedding Layer
                self._embedding_layer = OnDeviceEmbedding(
                    vocab_size=self.vocab_size,
                    embedding_width=self.embedding_size,
                    initializer=initializer,
                    name="word_embeddings",
                )
            if self.use_type_embeddings:
                # Type Embeddings
                self._type_embeddings = OnDeviceEmbedding(
                    vocab_size=self.type_vocab_size,
                    embedding_width=self.embedding_size,
                    initializer=initializer,
                    name="type_embeddings",
                )
            if self.use_positonal_embeddings:
                # Positional Embedding
                self._position_embedding_layer = SimplePositionEmbedding(
                    initializer=initializer,
                    max_sequence_length=self.max_position_embeddings,
                    embedding_width=self.embedding_size,
                    name="positional_embeddings",
                )

        else:
            # Word Embedding Layer
            self._embedding_layer = OnDeviceEmbedding(
                vocab_size=self.vocab_size,
                embedding_width=self.embedding_size,
                initializer=initializer,
                name="word_embeddings",
            )

            if self.use_type_embeddings:
                # Type Embeddings
                self._type_embeddings = OnDeviceEmbedding(
                    vocab_size=self.type_vocab_size,
                    embedding_width=self.embedding_size,
                    initializer=initializer,
                    name="type_embeddings",
                )
            if self.use_positonal_embeddings:
                # Positional Embedding
                self._position_embedding_layer = SimplePositionEmbedding(
                    initializer=initializer,
                    max_sequence_length=self.max_position_embeddings,
                    embedding_width=self.embedding_size,
                    name="positional_embeddings",
                )

        # Embedding Norm
        self._embedding_norm = tf.keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=self.layer_norm_epsilon,
            dtype=tf.float32,
        )

        # Embedding dropout Layer
        self._embedding_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)

        # Transformer Layer
        self._transformer_layers = []
        for i in range(self.num_hidden_layers):
            layer = TransformerBERT(
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                intermediate_activation=self.activation,
                dropout_rate=self.hidden_dropout_prob,
                attention_dropout_rate=self.attention_probs_dropout_prob,
                kernel_initializer=self.initializer,
                is_training=self.is_training,
                use_dropout=self.use_dropout,
                is_decoder=is_decoder,
                layer_norm_epsilon=self.layer_norm_epsilon,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        self._pooler_layer = tf.keras.layers.Dense(
            units=self.embedding_size,
            activation="tanh",
            kernel_initializer=self.initializer,
            name="pooler_transform",
        )

        self.mlm_layer = MLMLayer(
            self.embedding_size,
            self.layer_norm_epsilon,
            self.hidden_act,
            name="mlm_layer",
        )

        self._last_logits_bias = self.add_weight(
            "tf_transformers/last_logits_bias",
            shape=(self.vocab_size,),
            dtype=tf.float32,
            trainable=True,
        )

        # Initialize model
        self.model_inputs = self.get_model(initialize_only=True)
        logging.info("Initialized Variables")

        if self.model_dir:
            self.load_model(self, self.model_dir)
            logging.info("Loaded Variables from {}".format(self.model_dir))

    def call_predict(self, inputs):
        """Inputs will be pass to this method, when is_training = False.
        The need to cache the past `key` and `value` \
        tensors are necessary while predicting, to make the inference/NLG
        faster in case of AutoRegressive Decoding.

        """
        input_ids_mod = inputs["input_ids"]
        all_cache_key = inputs["all_cache_key"]
        all_cache_value = inputs["all_cache_value"]
        past_length = inputs["past_length"]

        # Come from kwargs
        if self.mask_mode in ["user_defined", "prefix"]:
            input_mask = inputs["input_mask"]
        if self.use_type_embeddings:
            input_type_ids = inputs["input_type_ids"]

        # Convert past_length 2D to 1D
        past_length = tf.squeeze(past_length, 0)

        # In case of variable batch decoding, we will pad the inputs with -1
        # So, we will replace -1 with 0, because -1 is not a valid index in word embeddings
        # >> input_ids_mod = [[ 1, 5, 7,  8,  10],
        #                       2, 3, -1, -1, -1]]
        #
        # >> input_ids     = [[1, 5, 7, 8,10],
        #                      2, 3, 0, 0, 0]]

        input_ids = input_ids_mod * tf.cast(tf.not_equal(input_ids_mod, -1), tf.int32)
        sequence_length = tf.shape(input_ids)[1]

        # Asserting
        tf.assert_equal(tf.shape(all_cache_value)[0], self.num_hidden_layers)

        # Step 0 of inference. For step0, we do not have valid cache. We pass zero tensor
        def step_0(input_ids):
            sequence_length = tf.shape(input_ids)[1]
            position_embeddings = self._position_embedding_layer(tf.range(sequence_length))
            return sequence_length, position_embeddings

        # From step_1 (autoregressive mode starts) onwards, we need to account for
        # `past_length` of previous words (inputs + generated) . Due to our logic,
        # we need to take a transpose of `position_embeddings` in this specific setting
        def step_other(input_ids):
            sequence_length = tf.shape(input_ids)[1]
            # Because past_length varies with batch
            position_embeddings = self._position_embedding_layer(past_length + sequence_length)
            position_embeddings = tf.transpose(position_embeddings, [1, 0, 2])
            return sequence_length, position_embeddings

        # Condition to switch functions
        # if `sum(past_length) = 0` , means no outputs has been generated. the given inputs is the first input # noqa
        sequence_length, positional_embeddings = tf.cond(
            tf.equal(tf.reduce_sum(past_length), 0),
            lambda: step_0(input_ids),
            lambda: step_other(input_ids),
        )

        all_cache_key = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_key, num_or_size_splits=self.num_hidden_layers, axis=0)
        ]
        all_cache_value = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_value, num_or_size_splits=self.num_hidden_layers, axis=0)
        ]

        word_embeddings = self._embedding_layer(input_ids)
        embeddings = word_embeddings
        # Add word_embeddings + position_embeddings + type_embeddings
        if self.use_type_embeddings:
            type_embeddings = self._type_embeddings(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self.use_positonal_embeddings:
            embeddings = embeddings + positional_embeddings

        # Norm + dropout
        embeddings = self._embedding_norm(embeddings)
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)

        # Initialize `attention_mask` as empty list
        attention_mask = []
        if self.mask_mode == "user_defined":
            attention_mask = SelfAttentionMask()([embeddings, input_mask])
        if self.mask_mode == "prefix":
            attention_mask = tf.map_fn(prefix_mask, input_mask, fn_output_signature=tf.float32)
        if self.mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        encoder_outputs = []
        # Make all -1 positions to 0 (as -1 represents padding in the input)
        mask_values = tf.cast(tf.not_equal(input_ids_mod, -1), tf.float32)
        # We want zero values , where embeddings inputs where 0 (by replacing PAD -1)
        # So we use the mask and multiply it with embeddings
        embeddings = embeddings * tf.expand_dims(mask_values, -1)
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            # Fetching
            cache_value = all_cache_value[i]
            cache_key = all_cache_key[i]

            embeddings, cache_key, cache_value = layer(
                [embeddings, attention_mask],
                cache_key=cache_key,
                cache_value=cache_value,
            )
            # Updating
            all_cache_key[i] = cache_key
            all_cache_value[i] = cache_value

            # Mask next layer embedding (PAD positions to 0)
            embeddings = tf.identity(
                embeddings * tf.expand_dims(mask_values, -1),
                name="encoder_outputs_{}".format(i),
            )
            encoder_outputs.append(embeddings)

        def step_0_gather(past_length):
            cache_length = tf.reduce_sum(tf.cast(tf.not_equal(input_ids_mod, -1), tf.int32), axis=1) - 1
            # Getting corresponding last token tensor and last token logits
            last_token_tensor = tf.gather_nd(embeddings, tf.expand_dims(cache_length, axis=1), batch_dims=1)
            past_length = past_length + cache_length
            return past_length, last_token_tensor

        def step_other_gather(past_length):
            past_length = past_length + sequence_length
            last_token_tensor = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(encoder_outputs[-1])
            return past_length, last_token_tensor

        # First word of last layer outputs [CLS]
        cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(encoder_outputs[-1])
        # batch_size x embedding_size
        cls_output = self._pooler_layer(cls_token_tensor)

        # batch_size x sequence_length x embedding_size
        token_embeddings = encoder_outputs[-1]

        # Condition to switch functionsn (When batch_size > 1, past_length will be different for each entry) # noqa
        # if `sum(past_length) = 0` , means no outputs has been generated. the given inputs is the first input # noqa
        past_length, last_token_tensor = tf.cond(
            tf.equal(tf.reduce_sum(past_length), 0),
            lambda: step_0_gather(past_length),
            lambda: step_other_gather(past_length),
        )

        # unilm has one
        token_embeddings_extra = self.mlm_layer(token_embeddings)

        # token --> vocab ( batch_size x sequence_length x vocab_size)
        token_logits = (
            tf.matmul(
                token_embeddings_extra,
                self.get_embedding_table(),
                transpose_b=True,
                name="token_logits",
            )
            + self._last_logits_bias
        )

        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        # Expand dims of past_length back to 2D
        past_length = tf.expand_dims(past_length, 0, name="past_length")
        # Stack all layers key and value together
        # num_layers x batch_size x num_heads x sequence_length x (hidden_dimension/num_heads)
        all_cache_key = tf.stack(all_cache_key, axis=0, name="all_cache_key")
        all_cache_value = tf.stack(all_cache_value, axis=0, name="all_cache_value")

        return {
            "cls_output": cls_output,
            "last_token_logits": last_token_logits,
            "past_length": past_length,
            "all_cache_key": all_cache_key,
            "all_cache_value": all_cache_value,
        }

    def call_training(self, inputs):
        """Forward Pass for BERT

        Args:
            inputs: dict
            inputs is a dict with keys  [`input_ids` , `input_mask`, `input_type_ids`].
            These keys might or might not be present based on `mask_mode` and other criterias

        """
        input_ids = inputs["input_ids"]
        # When `mask_mode` is `causal` , input_mask is not required
        if self.mask_mode in ["user_defined", "prefix"]:
            input_mask = inputs["input_mask"]
        # Default True in BERT
        if self.use_type_embeddings:
            input_type_ids = inputs["input_type_ids"]

        sequence_length = tf.shape(input_ids)[1]
        word_embeddings = self._embedding_layer(input_ids)
        embeddings = word_embeddings
        # Add word_embeddings + position_embeddings + type_embeddings
        if self.use_type_embeddings:
            type_embeddings = self._type_embeddings(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self.use_positonal_embeddings:
            positional_embeddings = self._position_embedding_layer(tf.range(sequence_length))
            embeddings = embeddings + positional_embeddings

        # Norm + dropout
        embeddings = self._embedding_norm(embeddings)
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)

        # Initialize `attention_mask` as empty list
        attention_mask = []
        if self.mask_mode == "user_defined":
            attention_mask = SelfAttentionMask()([embeddings, input_mask])
        if self.mask_mode == "prefix":
            attention_mask = tf.map_fn(prefix_mask, input_mask, dtype=tf.float32)
        if self.mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        encoder_outputs = []
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            embeddings, _, _ = layer([embeddings, attention_mask])
            encoder_outputs.append(embeddings)

        # First word of last layer outputs [CLS]
        cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(encoder_outputs[-1])
        # batch_size x embedding_size
        cls_output = self._pooler_layer(cls_token_tensor)
        # batch_size x sequence_length x embedding_size
        token_embeddings = encoder_outputs[-1]

        # unilm has one
        token_embeddings_extra = self.mlm_layer(token_embeddings)

        # token --> vocab ( batch_size x sequence_length x vocab_size)
        token_logits = (
            tf.matmul(
                token_embeddings_extra,
                self.get_embedding_table(),
                transpose_b=True,
                name="token_logits",
            )
            + self._last_logits_bias
        )

        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        return {
            "cls_output": cls_output,
            "token_embeddings": token_embeddings_extra,
            "all_layer_token_embeddings": encoder_outputs,
            "token_logits": token_logits,
            "last_token_logits": last_token_logits,
        }

    def call_decoder_predict(self, inputs):
        """Inputs will be pass to this method, when is_training = False and is_decoder = True.
        The need to cache the past `key` and `value` tensors for decoders necessary \
         while predicting, to make the inference/NLG
        faster in case of AutoRegressive Decoding.

        """

        input_ids = inputs["input_ids"]
        encoder_hidden_state = inputs["encoder_hidden_states"]
        decoder_encoder_mask = inputs["decoder_encoder_mask"]
        all_cache_key = inputs["all_cache_key"]
        all_cache_value = inputs["all_cache_value"]

        # Decoder don't need this

        # # When `mask_mode` is `causal` , input_mask is not required
        # if self.mask_mode in ['user_defined']:
        #     input_mask     = inputs['input_mask']

        if self.use_type_embeddings:
            input_type_ids = inputs["input_type_ids"]

        # cache_length = tf.constant(0, dtype=tf.int32)

        def step_0_cache_length(_):
            return tf.constant(0, dtype=tf.int32)

        def step_other_cache_length(all_cache_key):
            past_length = tf.shape(all_cache_key)[3]
            # Why -1, because When iter 2 (our positional embedding should be 1 not 2 and so on)
            sequence_length = tf.shape(input_ids)[1] + past_length - 1
            return sequence_length

        sequence_length = tf.cond(
            tf.equal(tf.reduce_sum(all_cache_key), 0),
            lambda: step_0_cache_length(all_cache_key),
            lambda: step_other_cache_length(all_cache_key),
        )

        all_cache_key = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_key, num_or_size_splits=self.num_hidden_layers, axis=0)
        ]
        all_cache_value = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_value, num_or_size_splits=self.num_hidden_layers, axis=0)
        ]

        # If decoder is not sharing embeddings
        if self.initialize_embeddings:
            word_embeddings = self._embedding_layer(input_ids)
            embeddings = word_embeddings
            # Add word_embeddings + position_embeddings + type_embeddings
            if self.use_type_embeddings:
                type_embeddings = self._type_embeddings(input_type_ids)
                embeddings = embeddings + type_embeddings
            if self.use_positonal_embeddings:
                positional_embeddings = self._position_embedding_layer(sequence_length)
                # Make it 3D for sum ( For decoder we decode one at a time)
                positional_embeddings = tf.expand_dims(positional_embeddings, 0)
                embeddings = embeddings + positional_embeddings
        else:
            embeddings = inputs["decoder_embeddings"]
        # Norm + dropout
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)

        # Initialize `attention_mask` as empty list
        attention_mask = []
        if self.mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        decoder_outputs = []
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            # Fetching
            cache_value = all_cache_value[i]
            cache_key = all_cache_key[i]

            embeddings, cache_key, cache_value = layer(
                [
                    embeddings,
                    attention_mask,
                    encoder_hidden_state,
                    decoder_encoder_mask,
                ],
                cache_key=cache_key,
                cache_value=cache_value,
            )

            # Updating
            all_cache_key[i] = cache_key
            all_cache_value[i] = cache_value

            decoder_outputs.append(embeddings)

        # Stack all layers key and value together
        # num_layers x batch_size x num_heads x sequence_length x (hidden_dimension/num_heads)
        all_cache_key = tf.stack(all_cache_key, axis=0, name="all_cache_key")
        all_cache_value = tf.stack(all_cache_value, axis=0, name="all_cache_value")

        # batch_size x sequence_length x embedding_size
        token_embeddings = decoder_outputs[-1]

        return {
            "all_cache_key": all_cache_key,
            "all_cache_value": all_cache_value,
            "token_embeddings": token_embeddings,
            "all_layer_token_embeddings": decoder_outputs,
        }

    def call_decoder(self, inputs):
        """Forward Pass for Decoder

        Args:
            inputs: dict
            inputs is a dict with keys  [`input_ids` , `input_mask`, `input_type_ids`, \
             `encoder_hidden_states`, `decoder_encoder_mask`].
            These keys might or might not be present based on `mask_mode` and other criterias

        """
        input_ids = inputs["input_ids"]
        encoder_output = inputs["encoder_hidden_states"]
        decoder_encoder_mask = inputs["decoder_encoder_mask"]

        if self.mask_mode in ["user_defined"]:
            input_mask = inputs["input_mask"]

        if self.use_type_embeddings:
            input_type_ids = inputs["input_type_ids"]

        sequence_length = tf.shape(input_ids)[1]

        # If decoder is not sharing embeddings
        if self.initialize_embeddings:
            word_embeddings = self._embedding_layer(input_ids)
            embeddings = word_embeddings
            # Add word_embeddings + position_embeddings + type_embeddings
            if self.use_type_embeddings:
                type_embeddings = self._type_embeddings(input_type_ids)
                embeddings = embeddings + type_embeddings
            if self.use_positonal_embeddings:
                positional_embeddings = self._position_embedding_layer(tf.range(sequence_length))
                embeddings = embeddings + positional_embeddings
        else:
            embeddings = inputs["decoder_embeddings"]
        # Norm + dropout
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)
        # Initialize `attention_mask` as empty list
        attention_mask = []

        if self.mask_mode == "user_defined":
            attention_mask = SelfAttentionMask()([embeddings, input_mask])
        if self.mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)
        decoder_outputs = []
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            embeddings, _key, _value = layer([embeddings, attention_mask, encoder_output, decoder_encoder_mask])

            decoder_outputs.append(embeddings)

        # batch_size x sequence_length x embedding_size
        token_embeddings = decoder_outputs[-1]
        return {
            "token_embeddings": token_embeddings,
            "all_layer_token_embeddings": decoder_outputs,
        }

    def call(self, inputs):
        """Forward Pass.
        We have 2 pipelines . Training pipeline is relatively simpler
        Testing pipeline has few changes to accomodate caching of `key` and `value` for Transformer.
        Caching is significant for AutoRegressive modeling.
        Also, minor changes to make use of variable batch decoding

        Args: inputs, dict

            if self.is_training:
                self.call_training(inputs)
            else:
                self.call_predict(inputs)

        """
        # Training Pipeline
        if self.is_training:
            # Decoder Mode Training
            if self.is_decoder:
                outputs = self.call_decoder(inputs)
            # Encoder Mode / Mode for anything except Decoder
            else:
                # Note: This mode is also be used with `use_dropout`
                # is False, when `is_training` is False and `pipeline_mode` is None
                # Default (Training/ Predict) pipeline
                outputs = self.call_training(inputs)

        # Predict Pipeline ( For decoder and auto-regressive mode only)
        else:
            # Decoder Predict pipeline
            if self.is_decoder:
                if self.pipeline_mode == "auto-regressive":
                    outputs = self.call_decoder_predict(inputs)
                else:
                    # is_training with use_dropout = False
                    outputs = self.call_decoder(inputs)
            else:
                # Encoder Model (Predict pipeline) All models can be in this form
                if self.pipeline_mode == "auto-regressive":
                    outputs = self.call_predict(inputs)

        return outputs

    def get_and_load_model(self, model_dir=None, **kwargs):
        """Convert tf.keras.Layer to tf.keras.Model
        Load the model from checkpoint folder (.ckpt)

        """
        # get keras model from keras layer
        model_obj = self.get_model()
        if model_dir:
            # Load the model using checkpoint
            self.load_model(model_obj, model_dir, kwargs)
        return model_obj

    def show_sample_model_inputs_outputs(self):
        """Show the sample inputs , so that users won't get confused with various settings"""
        sample_inputs = self.get_model().inputs
        sample_outputs = self.get_model().outputs
        return {"sample_inputs": sample_inputs, "sample_outputs": sample_outputs}

    def extend_positional_embeddings(self, factor):
        """Extends positional embeddings, by a factor.
        If factor = 2, we replicate the positional embeddings. \
        If matrix is 512 x 768 , we convert it into 1024 x 768.

        Args:
            factor: int

        Returns:
            a new object of the class method
        """
        if not isinstance(factor, int):
            raise ValueError(" `factor` must be an int with value > 1")
        # Squeeze is used to convert 3D to 2D
        updated_pos_embeddings = tf.squeeze(tf.repeat(self._position_embedding_layer.variables, factor, axis=1), 0)

        self.config["max_position_embeddings"] = 2 * self.config["max_position_embeddings"]

        tf.keras.backend.clear_session()
        new_layer = self.__class__(
            config=self.config,
            mask_mode=self.mask_mode,
            name=self.model_name,
            use_dropout=self.use_dropout,
            is_training=self.is_training,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            use_type_embeddings=self.use_type_embeddings,
            pipeline_mode=self.pipeline_mode,
        )

        # layer to model to instantiate variables
        new_model = new_layer.get_model()
        del new_model

        model_new_dict = {}
        for var in self.variables:
            if "positional_embedding" in var.name:
                # Add the replicated previous embeddings to this embeddings
                model_new_dict[var.name] = updated_pos_embeddings
            else:
                model_new_dict[var.name] = var

        # Re assign it to model_new
        for var in new_layer.variables:
            var.assign(model_new_dict[var.name])
        # Release the memory
        del model_new_dict
        logging.info("Succesfully changed position_embeddings to {}".format(updated_pos_embeddings.shape))
        return new_layer

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def get_config(self):
        return self._config_dict

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
