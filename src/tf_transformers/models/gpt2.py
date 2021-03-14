# pylint: disable=g-classes-have-attributes
# from __future__ import google_type_annotations
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from absl import logging

from tf_transformers.activations import get_activation
from tf_transformers.core import LegacyLayer
from tf_transformers.layers import (GPT2LayerNormalization, MLMLayer,
                                    OnDeviceEmbedding, SimplePositionEmbedding)
from tf_transformers.layers.mask import (CausalMask, CrossAttentionMask,
                                         SelfAttentionMask, prefix_mask)
from tf_transformers.layers.transformer import TransformerGPT2

logging.set_verbosity("INFO")


class GPT2Encoder(LegacyLayer):
    """GPT2 based encoder / Decoder .
    Language Models are Unsupervised Multitask Learners
    Authors: Alec Radford , Jeffrey Wu , Rewon Child ,
            David Luan , Dario Amodei ,Ilya Sutskever

    Implementation of GPT2 in TF2.0

    Paper: https://arxiv.org/abs/1810.04805
    Official Code: https://github.com/openai/gpt-2


    """

    def __init__(
        self,
        config,
        mask_mode="causal",
        name=None,
        use_dropout=False,
        is_training=None,
        batch_size=None,
        sequence_length=None,
        use_type_embeddings=False,
        use_positonal_embeddings=True,
        pipeline_mode=None,
        is_decoder=False,
        cross_attention_inside_encoder=False,
        share_attention_layers=True,
        share_encoder_embeddings=False,
        encoder_embedding_layer=None,
        encoder_type_embedding_layer=None,
        encoder_positional_embedding_layer=None,
        use_mlm_layer=False,
        return_all_layer_token_embeddings=True,
        **kwargs,
    ):
        """
        Args:
            config: dict
            mask_mode: str, `user_defined` BERT by default uses masking for PADDED or MLM. But can be overridden . # noqa
            name: str, Name of the model
            use_dropout: bool, It is strictly optional. Sometimes,
                        while training you can set `use_dropout` to False.
                         If `is_training` is False, `use_dropout` will be automatically set to False. # noqa
            batch_size: int, `batch_size` can be None or any int
            sequence_length: int, `sequence_length` can be None or any int
            use_type_embeddings: bool, By default BERT has type_embeddings, GPT2 don't.
            use_positonal_embeddings: bool, T5 don't have postional embeddings
            bidirectional: use in relative postional embedding (we can infer it based on mask_mode)
            is_decoder: bool, if True it will become decoder mode (as in Seq2Seq)
            use_mlm_layer: bool ( To use MLM layer or not )
            share_encoder_embeddings: bool, When is_decoder = True, most cases,
                            it will re-use the embedding layer from Encoder.
                            So. if you still want to initialize , set this to False.
                            If True, share embedding layers from encoder
                            (word_embeddings, positional_embeddings, type_embeddings)
            cross_attention_inside_encoder: bool, Encoder Decoder Cross attention in each layer
        """
        # Because saved_model causes some serialization problems here
        # self.config              = config
        self.vocab_size = config["vocab_size"]
        self.type_vocab_size = config["type_vocab_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = config["attention_head_size"]
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

        # self._self_setattr_tracking = False
        self.mask_mode = mask_mode
        self.use_dropout = use_dropout
        self.is_training = is_training
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.use_type_embeddings = use_type_embeddings
        self.use_positonal_embeddings = use_positonal_embeddings
        self.share_encoder_embeddings = share_encoder_embeddings
        self.share_attention_layers = share_attention_layers
        self.use_mlm_layer = use_mlm_layer
        self.cross_attention_inside_encoder = cross_attention_inside_encoder
        self.return_all_layer_token_embeddings = return_all_layer_token_embeddings

        if not name.startswith("tf_transformers"):
            kwargs["name"] = "tf_transformers/" + self.model_name
        else:
            kwargs["name"] = self.model_name
        self.validate_and_set_inputs()

        super(GPT2Encoder, self).__init__(is_training=self.is_training, use_dropout=self.use_dropout, **kwargs)
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
            "share_encoder_embeddings": self.share_encoder_embeddings,
            "share_attention_layers": self.share_attention_layers,
            "cross_attention_inside_encoder": cross_attention_inside_encoder,
            "return_all_layer_token_embeddings": self.return_all_layer_token_embeddings,
        }

        # Update config dict with passed config
        self._config_dict.update(config)

        # Call embedding layers
        self._embedding_layer, self._type_embeddings, self._position_embedding_layer = self.get_embedding_layers()
        if self.is_decoder:
            # If embedding has to shared from the encoder
            if self.share_encoder_embeddings:
                self._embedding_layer = encoder_embedding_layer
                self._type_embeddings = encoder_type_embedding_layer
                self._position_embedding_layer = encoder_positional_embedding_layer

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
            layer = TransformerGPT2(
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                intermediate_activation=self.activation,
                dropout_rate=self.hidden_dropout_prob,
                attention_dropout_rate=self.attention_probs_dropout_prob,
                kernel_initializer=self.initializer,
                is_training=self.is_training,
                use_dropout=self.use_dropout,
                is_decoder=is_decoder,
                share_attention_layers=share_attention_layers,
                layer_norm_epsilon=self.layer_norm_epsilon,
                cross_attention_inside_encoder=self.cross_attention_inside_encoder,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        if self.use_mlm_layer:
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
        # Last Layer Normalization (only in GPT2)
        self._last_layer_norm = GPT2LayerNormalization(
            name="ln_f/layer_norm",
            axis=-1,
            epsilon=self.layer_norm_epsilon,
            dtype=tf.float32,
        )
        self.call_fn = self.get_call_method()
        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)
        logging.info("Initialized Variables")

    def call_predict(self, inputs):
        """Inputs will be pass to this method, when is_training = False.
        The need to cache the past `key` and `value` tensors are \
        necessary while predicting, to make the inference/NLG
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
        # So, we will replace -1 with 0, because -1 \
        # is not a valid index in word embeddings
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
        # if `sum(past_length) = 0` , means no outputs has been generated. \
        # the given inputs is the first input
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

        def step_0_gather(past_length, token_embeddings):
            cache_length = tf.reduce_sum(tf.cast(tf.not_equal(input_ids_mod, -1), tf.int32), axis=1) - 1
            # Getting corresponding last token tensor and last token logits
            last_token_tensor = tf.gather_nd(token_embeddings, tf.expand_dims(cache_length, axis=1), batch_dims=1)
            past_length = past_length + cache_length
            return past_length, last_token_tensor

        def step_other_gather(past_length, token_embeddings):
            past_length = past_length + sequence_length
            last_token_tensor = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_embeddings)
            return past_length, last_token_tensor

        # batch_size x sequence_length x embedding_size
        token_embeddings = self._last_layer_norm(encoder_outputs[-1])

        # Condition to switch functionsn (When batch_size > 1,
        # past_length will be different for each entry)
        # if `sum(past_length) = 0` , means no outputs has been generated.
        # the given inputs is the first input
        past_length, last_token_tensor = tf.cond(
            tf.equal(tf.reduce_sum(past_length), 0),
            lambda: step_0_gather(past_length, token_embeddings),
            lambda: step_other_gather(past_length, token_embeddings),
        )

        # token --> vocab ( batch_size x sequence_length x vocab_size)
        last_token_logits = tf.matmul(
            last_token_tensor,
            self.get_embedding_table(),
            transpose_b=True,
            name="token_logits",
        )
        # last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        # Expand dims of past_length back to 2D
        past_length = tf.expand_dims(past_length, 0, name="past_length")
        # Stack all layers key and value together
        # num_layers x batch_size x num_heads x sequence_length x (hidden_dimension/num_heads)
        all_cache_key = tf.stack(all_cache_key, axis=0, name="all_cache_key")
        all_cache_value = tf.stack(all_cache_value, axis=0, name="all_cache_value")

        return {
            "token_embeddings": token_embeddings,
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
            embeddings = tf.identity(embeddings, name="token_embeddings_layer_{}".format(i))
            encoder_outputs.append(embeddings)

        # Last layer output has to be normalized in GPT2
        encoder_outputs[-1] = self._last_layer_norm(encoder_outputs[-1])
        # batch_size x sequence_length x embedding_size
        token_embeddings = encoder_outputs[-1]

        # token --> vocab ( batch_size x sequence_length x vocab_size)
        token_logits = tf.matmul(
            token_embeddings,
            self.get_embedding_table(),
            transpose_b=True,
            name="token_logits",
        )
        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)
        last_token_logits = tf.identity(last_token_logits, name="last_token_logits")

        result = {
            "token_embeddings": token_embeddings,
            "token_logits": token_logits,
            "last_token_logits": last_token_logits,
        }
        if self.return_all_layer_token_embeddings:
            result["all_layer_token_embeddings"] = encoder_outputs
        return result

    def call_cross_attention_encoder(self, inputs):
        """[summary]

        Args:
            inputs ([type]): [description]
        """
        encoder_input_ids = inputs["encoder_input_ids"]
        decoder_input_ids = inputs["decoder_input_ids"]
        encoder_input_type_ids = None
        decoder_input_type_ids = None

        if self.use_type_embeddings:
            encoder_input_type_ids = inputs["encoder_input_type_ids"]
            decoder_input_type_ids = inputs["decoder_input_type_ids"]
        encoder_input_mask = None
        if self.mask_mode in ["user_defined", "prefix"]:
            encoder_input_mask = inputs["encoder_input_mask"]

        def get_embeddings(input_ids, input_type_ids):
            """Get embedding for encoder as well as decoder

            Args:
                input_ids ([type]): [description]
                input_type_ids ([type]): [description]
            """

            embeddings = self._embedding_layer(input_ids)
            sequence_length = tf.shape(input_ids)[1]
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
            return embeddings

        encoder_embeddings = get_embeddings(encoder_input_ids, encoder_input_type_ids)
        decoder_embeddings = get_embeddings(decoder_input_ids, decoder_input_type_ids)

        # Initialize `encoder_attention_mask` as empty list
        encoder_attention_mask = []
        if self.mask_mode == "user_defined":
            encoder_attention_mask = SelfAttentionMask()([encoder_embeddings, encoder_input_mask])
        if self.mask_mode == "prefix":
            encoder_attention_mask = tf.map_fn(prefix_mask, encoder_input_mask, dtype=tf.float32)
        if self.mask_mode == "causal":
            encoder_attention_mask = CausalMask()(encoder_embeddings)

        # Decoder mask is always None
        decoder_attention_mask = CausalMask()(decoder_embeddings)
        decoder_encoder_mask = CrossAttentionMask()([decoder_input_ids, encoder_input_mask])
        decoder_outputs = []
        encoder_outputs = []

        # Encoder Layer
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            encoder_embeddings, _, _ = layer(
                [
                    encoder_embeddings,
                    encoder_attention_mask,
                    decoder_encoder_mask,  # dummy decoder_encoder_mask
                    encoder_embeddings,  # dummy encoder_hidden_states
                ],
                mode="encoder",
            )
            encoder_outputs.append(encoder_embeddings)

        # Decoder Layer
        encoder_hidden_states = encoder_outputs[-1]
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            decoder_embeddings, _, _ = layer(
                [decoder_embeddings, decoder_attention_mask, decoder_encoder_mask, encoder_hidden_states],
                mode="decoder",
            )
            decoder_outputs.append(decoder_embeddings)

        decoder_outputs[-1] = self._last_layer_norm(decoder_outputs[-1])
        # First word of last layer outputs [CLS]
        # cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(decoder_outputs[-1])
        # batch_size x embedding_size
        # cls_output = self._pooler_layer(cls_token_tensor)
        # batch_size x sequence_length x embedding_size
        token_embeddings = decoder_outputs[-1]

        # MLM Projection
        if self.use_mlm_layer:
            token_embeddings = self.mlm_layer(token_embeddings)
            # token --> vocab ( batch_size x sequence_length x vocab_size)
            token_logits = (
                tf.matmul(
                    token_embeddings,
                    self.get_embedding_table(),
                    transpose_b=True,
                    name="token_logits",
                )
                + self._last_logits_bias
            )
        else:

            # token --> vocab ( batch_size x sequence_length x vocab_size)
            token_logits = tf.matmul(
                token_embeddings,
                self.get_embedding_table(),
                transpose_b=True,
                name="token_logits",
            )

        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        result = {
            "token_embeddings": token_embeddings,
            "token_logits": token_logits,
            "last_token_logits": last_token_logits,
        }

        if self.return_all_layer_token_embeddings:
            result["all_layer_token_embeddings"] = decoder_outputs
        return result

    def call_cross_attention_encoder_predict(self, inputs):
        """[summary]

        Args:
            inputs ([type]): [description]
        """

        encoder_input_ids = inputs["encoder_input_ids"]
        decoder_input_ids = inputs["decoder_input_ids"]
        encoder_input_type_ids = None
        decoder_input_type_ids = None

        if self.use_type_embeddings:
            encoder_input_type_ids = inputs["encoder_input_type_ids"]
            decoder_input_type_ids = inputs["decoder_input_type_ids"]
        encoder_input_mask = None
        if self.mask_mode in ["user_defined", "prefix"]:
            encoder_input_mask = inputs["encoder_input_mask"]

        # self.num_hidden_layers, batch_size, sequence_length, embeddingd_imension
        encoder_hidden_states = inputs["encoder_hidden_states"]
        all_cache_key = inputs["decoder_all_cache_key"]
        all_cache_value = inputs["decoder_all_cache_value"]

        def get_encoder_embeddings(input_ids, input_type_ids):
            """Get embedding for encoder as well as decoder

            Args:
                input_ids ([type]): [description]
                input_type_ids ([type]): [description]
            """
            embeddings = self._embedding_layer(input_ids)
            sequence_length = tf.shape(input_ids)[1]
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
            return embeddings

        # this function is slightly different from the other function
        # because, we do not need tf.range(sequence_length)
        # we need it for (one word) from, step 1 onwards, as we decode
        # word by word. So we use all_cache_key for getting the past_length

        def get_decoder_embeddings_step_other(input_ids, input_type_ids):
            """Get embedding for encoder as well as decoder

            Args:
                input_ids ([type]): [description]
                input_type_ids ([type]): [description]
            """

            def step_0_cache_length(_):
                return tf.constant(0, dtype=tf.int32)

            def step_other_cache_length(all_cache_key):
                past_length = tf.shape(all_cache_key)[3]
                # Why -1, because When iter 2 (our positional
                # embedding should be 1 not 2 and so on)
                sequence_length = tf.shape(input_ids)[1] + past_length - 1
                return sequence_length

            sequence_length = tf.cond(
                tf.equal(tf.reduce_sum(all_cache_key), 0),
                lambda: step_0_cache_length(all_cache_key),
                lambda: step_other_cache_length(all_cache_key),
            )

            embeddings = self._embedding_layer(input_ids)
            # Add word_embeddings + position_embeddings + type_embeddings
            if self.use_type_embeddings:
                type_embeddings = self._type_embeddings(input_type_ids)
                embeddings = embeddings + type_embeddings
            if self.use_positonal_embeddings:
                positional_embeddings = self._position_embedding_layer(sequence_length)
                # Make it 3D for sum ( For decoder we decode one at a time)
                positional_embeddings = tf.expand_dims(positional_embeddings, 0)
                embeddings = embeddings + positional_embeddings
            # Norm + dropout
            embeddings = self._embedding_norm(embeddings)
            embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)
            return embeddings

        # Encoder embeddings remains same throughout the decoding process
        # so we have to calculate it only once
        # So , we check if cache_key == 0, if its 0 its step 0
        # else, pass a dummy encoder_embeddings, as we dont have to use it from step1
        # because, what we need from encoder is encoder_hidden_states_batch

        encoder_embeddings = tf.cond(
            tf.equal(tf.reduce_sum(all_cache_key), 0.0),
            lambda: get_encoder_embeddings(encoder_input_ids, encoder_input_type_ids),
            lambda: tf.zeros_like(encoder_hidden_states),  # dummy
        )

        decoder_embeddings = tf.cond(
            tf.equal(tf.reduce_sum(all_cache_key), 0.0),
            lambda: get_encoder_embeddings(decoder_input_ids, decoder_input_type_ids),
            lambda: get_decoder_embeddings_step_other(decoder_input_ids, decoder_input_type_ids),
        )

        # Initialize `encoder_attention_mask` as empty list
        encoder_attention_mask = []
        if self.mask_mode == "user_defined":
            encoder_attention_mask = SelfAttentionMask()([encoder_embeddings, encoder_input_mask])
        if self.mask_mode == "prefix":
            encoder_attention_mask = tf.map_fn(prefix_mask, encoder_input_mask, dtype=tf.float32)
        if self.mask_mode == "causal":
            encoder_attention_mask = CausalMask()(encoder_embeddings)

        # Decoder mask is always None
        decoder_attention_mask = CausalMask()(decoder_embeddings)
        decoder_encoder_mask = CrossAttentionMask()([decoder_input_ids, encoder_input_mask])

        all_cache_key = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_key, num_or_size_splits=self.num_hidden_layers, axis=0)
        ]
        all_cache_value = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_value, num_or_size_splits=self.num_hidden_layers, axis=0)
        ]

        def calculate_encoder_hidden_state(encoder_embeddings):
            # Encoder Layer
            encoder_outputs = []
            for i in range(self.num_hidden_layers):
                layer = self._transformer_layers[i]
                cache_key = all_cache_key[i]
                cache_value = all_cache_value[i]
                encoder_embeddings, _, _ = layer(
                    [
                        encoder_embeddings,
                        encoder_attention_mask,
                        decoder_encoder_mask,  # decoder_encoder_mask
                        encoder_embeddings,
                    ],
                    mode="encoder",
                    cache_key=cache_key,
                    cache_value=cache_value,
                )
                encoder_outputs.append(encoder_embeddings)
            encoder_hidden_states = encoder_outputs[-1]
            return encoder_hidden_states

        # While decoding we have to calculate it only once
        def use_cache_encoder():
            return tf.identity(inputs["encoder_hidden_states"])

        encoder_hidden_states = tf.cond(
            tf.equal(tf.reduce_sum(inputs["encoder_hidden_states"]), 0.0),
            lambda: calculate_encoder_hidden_state(encoder_embeddings),
            lambda: use_cache_encoder(),
        )
        # Decoder layer
        decoder_outputs = []
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            # Fetching
            cache_value = all_cache_value[i]
            cache_key = all_cache_key[i]
            decoder_embeddings, cache_key, cache_value = layer(
                [
                    decoder_embeddings,
                    decoder_attention_mask,
                    decoder_encoder_mask,
                    encoder_hidden_states,
                ],
                mode="decoder",
                cache_key=cache_key,
                cache_value=cache_value,
            )
            # Updating
            all_cache_key[i] = cache_key
            all_cache_value[i] = cache_value
            decoder_outputs.append(decoder_embeddings)

        # Stack all layers key and value together
        # num_layers x batch_size x num_heads x sequence_length x
        # (hidden_dimension/num_heads) # noqa
        all_cache_key = tf.stack(all_cache_key, axis=0, name="decoder_all_cache_key")
        all_cache_value = tf.stack(all_cache_value, axis=0, name="decoder_all_cache_value")
        # First word of last layer outputs [CLS]
        # cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(decoder_outputs[-1])
        # batch_size x embedding_size
        # cls_output = self._pooler_layer(cls_token_tensor)
        # batch_size x sequence_length x embedding_size
        token_embeddings = decoder_outputs[-1]

        # MLM Projection
        if self.use_mlm_layer:
            token_embeddings = self.mlm_layer(token_embeddings)
            # token --> vocab ( batch_size x sequence_length x vocab_size)
            token_logits = (
                tf.matmul(
                    token_embeddings,
                    self.get_embedding_table(),
                    transpose_b=True,
                    name="token_logits",
                )
                + self._last_logits_bias
            )
        else:

            # token --> vocab ( batch_size x sequence_length x vocab_size)
            token_logits = tf.matmul(
                token_embeddings,
                self.get_embedding_table(),
                transpose_b=True,
                name="token_logits",
            )

        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)
        return {
            "encoder_hidden_states": encoder_hidden_states,
            "decoder_all_cache_key": all_cache_key,
            "decoder_all_cache_value": all_cache_value,
            "token_embeddings": token_embeddings,
            "token_logits": token_logits,
            "last_token_logits": last_token_logits,
        }

    def call_decoder_predict(self, inputs):
        """Inputs will be pass to this method, when is_training = False and is_decoder = True. # noqa
        The need to cache the past `key` and `value` tensors for decoders \
        necessary while predicting, to make the inference/NLG
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
            # Why -1, because When iter 2
            # (our positional embedding should be 1 not 2 and so on)
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
        decoder_outputs[-1] = self._last_layer_norm(decoder_outputs[-1])
        token_embeddings = decoder_outputs[-1]

        # token --> vocab ( batch_size x sequence_length x vocab_size)
        token_logits = tf.matmul(
            token_embeddings,
            self.get_embedding_table(),
            transpose_b=True,
            name="token_logits",
        )
        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        return {
            "all_cache_key": all_cache_key,
            "all_cache_value": all_cache_value,
            "token_embeddings": token_embeddings,
            "last_token_logits": last_token_logits,
        }

    def call_decoder(self, inputs):
        """Forward Pass for Decoder

        Args:
            inputs: dict
            inputs is a dict with keys  [`input_ids` , `input_mask`,
            `input_type_ids`, `encoder_hidden_states`, `decoder_encoder_mask`].
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
        decoder_outputs[-1] = self._last_layer_norm(decoder_outputs[-1])
        token_embeddings = decoder_outputs[-1]

        # token --> vocab ( batch_size x sequence_length x vocab_size)
        token_logits = tf.matmul(
            token_embeddings,
            self.get_embedding_table(),
            transpose_b=True,
            name="token_logits",
        )
        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        result = {
            "token_embeddings": token_embeddings,
            "token_logits": token_logits,
            "last_token_logits": last_token_logits,
        }

        if self.return_all_layer_token_embeddings:
            result["all_layer_token_embeddings"] = decoder_encoder_mask
        return result

    def call(self, inputs):
        """Forward Pass.
        We have 2 pipelines . Training pipeline is relatively simpler
        Testing pipeline has few changes to accomodate
        caching of `key` and `value` for Transformer.
        Caching is significant for AutoRegressive modeling.
        Also, minor changes to make use of variable batch decoding

        Args: inputs, dict

            if self.is_training:
                self.call_training(inputs)
            else:
                self.call_predict(inputs)

        """
        outputs = self.call_fn(inputs)
        return outputs

    def add_special_tokens(self, special_tokens=[], **kwargs):
        """
        In case of GPT2 special tokens are extending the original word embedding vocab,
        which means, we have to reshape the embedding layer and
        this wont be possible without some difficulty in
        TF2
        """
        if special_tokens == [] or None:
            return self.model_obj
        config_copy = self._config_dict.copy()
        config_copy["vocab_size"] = config_copy["vocab_size"] + len(special_tokens)

        tf.keras.backend.clear_session()
        new_model = self.__class__(config=config_copy, **kwargs)
        gpt2_model_new_dict = {}
        for var in self.variables:
            if "word_embeddings" in var.name:
                # emb_mean, emb_std = tf.reduce_mean(var), tf.math.reduce_std(var)
                nrow = len(special_tokens)
                # special_token_embedding = tf.random.normal(mean=emb_mean, stddev=emb_std,
                #                           seed = 1, shape=(nrow, self.embedding_dimension))
                # gpt2_model_new_dict[var.name]  = tf.concat([var, special_token_embedding], axis=0)

                special_token_embedding_layer = tf.keras.layers.Embedding(nrow, self.embedding_size)
                special_token_embedding = special_token_embedding_layer(tf.range(nrow))
                gpt2_model_new_dict[var.name] = tf.concat([var, special_token_embedding], axis=0)
            else:
                gpt2_model_new_dict[var.name] = var

        # Re assign it to model_new
        for var in new_model.variables:
            var.assign(gpt2_model_new_dict[var.name])
        # Release the memory
        del gpt2_model_new_dict

        return new_model

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
