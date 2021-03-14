from __future__ import absolute_import, division, print_function

import tensorflow as tf
from absl import logging

from tf_transformers.activations import get_activation
from tf_transformers.core import LegacyLayer
from tf_transformers.layers import (OnDeviceEmbedding, SimplePositionEmbedding,
                                    T5LayerNormalization)
from tf_transformers.layers.mask import (CausalMask, SelfAttentionMask,
                                         prefix_mask)
from tf_transformers.layers.transformer import TransformermT5

logging.set_verbosity("INFO")


class mT5Encoder(LegacyLayer):
    """T5 based encoder / Decoder .
    Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    Authors: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee,
             Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
    Implementation of T5 in TF2.0

    Paper: https://arxiv.org/abs/1910.10683
    Official Code: https://github.com/google-research/text-to-text-transfer-transformer


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
        use_type_embeddings=False,
        use_positonal_embeddings=False,
        pipeline_mode=None,
        is_decoder=False,
        share_encoder_embeddings=False,
        share_attention_layers=False,
        encoder_embedding_layer=None,
        encoder_type_embedding_layer=None,
        encoder_positional_embedding_layer=None,
        cross_attention_inside_encoder=False,
        return_all_layer_token_embeddings=True,
        **kwargs,
    ):
        """
        Args:
            config: dict
            mask_mode: str, `user_defined` BERT by default uses masking for PADDED or MLM. But can be overridden . # noqa
            name: str, Name of the model
            use_dropout: bool, It is strictly optional. Sometimes, while
                         training you can set `use_dropout` to False.
                         If `is_training` is False, `use_dropout` will be automatically set to False. # noqa
            batch_size: int, `batch_size` can be None or any int
            sequence_length: int, `sequence_length` can be None or any int
            use_type_embeddings: bool, By default BERT has type_embeddings, GPT2 don't.
            use_positonal_embeddings: bool, T5 don't have postional embeddings
            bidirectional: use in relative postional embedding (we can infer it based on mask_mode)
            is_decoder: bool, if True it will become decoder mode (as in Seq2Seq)
            share_encoder_embeddings: bool, When is_decoder = True, most cases,
                            it will re-use the embedding layer from Encoder.
                            So. if you still want to initialize , set this to False.
                            If True, share embedding layers from encoder
                            (word_embeddings, positional_embeddings, type_embeddings)
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
        self.bidirectional = config["bidirectional"]
        self.positional_buckets = config["positional_buckets"]
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

        self.cross_attention_inside_encoder = cross_attention_inside_encoder
        self.return_all_layer_token_embeddings = return_all_layer_token_embeddings

        if not name.startswith("tf_transformers"):
            kwargs["name"] = "tf_transformers/" + self.model_name
        else:
            kwargs["name"] = self.model_name
        self.validate_and_set_inputs()

        super(mT5Encoder, self).__init__(is_training=self.is_training, use_dropout=self.use_dropout, **kwargs)
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
        self._embedding_norm_layers = []

        # Embedding dropout Layer
        self._embedding_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)

        # Transformer Layer
        self._transformer_layers = []
        for i in range(self.num_hidden_layers):
            #  Required only for first layer to create the positonal_embeddings
            if i == 0:
                create_positonal_embedding = True
            else:
                create_positonal_embedding = False
            layer = TransformermT5(
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                attention_head_size=self.attention_head_size,
                intermediate_activation=self.intermediate_activation,
                bidirectional=self.bidirectional,
                create_positonal_embedding=create_positonal_embedding,
                positional_buckets=self.positional_buckets,
                dropout_rate=self.hidden_dropout_prob,
                attention_dropout_rate=self.attention_probs_dropout_prob,
                share_attention_layers=self.share_attention_layers,
                kernel_initializer=self.initializer,
                is_training=self.is_training,
                use_dropout=self.use_dropout,
                is_decoder=self.is_decoder,
                layer_norm_epsilon=self.layer_norm_epsilon,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        self._last_layer_norm = T5LayerNormalization(name="last_layer_norm", axis=-1, epsilon=1e-6, dtype=tf.float32)
        self._last_layer_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
        self.call_fn = self.get_call_method()
        # Initialize model
        self.model_inputs, self.model_ouputs = self.get_model(initialize_only=True)
        logging.info("Initialized Variables")

    def call_training(self, inputs):
        """Forward Pass for BERT

        Args:
            inputs: dict
            inputs is a dict with keys  [`input_ids` , `input_mask`, `input_type_ids`].
            These keys might or might not be
            present based on `mask_mode` and other criterias

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
            attention_mask = tf.map_fn(prefix_mask, input_mask, fn_output_signature=tf.float32)
        if self.mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        encoder_outputs = []
        position_bias = None
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            embeddings, position_bias, k, v = layer([embeddings, attention_mask], position_bias=position_bias)
            encoder_outputs.append(embeddings)

        encoder_outputs[-1] = self._last_layer_norm(encoder_outputs[-1])
        # batch_size x sequence_length x embedding_size
        token_embeddings = self._last_layer_dropout(encoder_outputs[-1])

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
            result["all_layer_token_embeddings"] = encoder_outputs
        return result

    def call_predict(self, inputs):
        """Inputs will be pass to this method, when is_training = False.
        The need to cache the past `key` and `value` tensors \
        are necessary while predicting, to make the inference/NLG
        faster in case of AutoRegressive Decoding.

        """

        raise NotImplementedError

    def call_decoder_predict(self, inputs):
        """Inputs will be pass to this method,
        when is_training = False and is_decoder = True.
        The need to cache the past `key` and `value` tensors for \
        decoders \necessary while predicting, to make the inference/NLG
        faster in case of AutoRegressive Decoding.

        """

        input_ids = inputs["input_ids"]
        encoder_hidden_state = inputs["encoder_hidden_states"]
        decoder_encoder_mask = inputs["decoder_encoder_mask"]
        all_cache_key = inputs["all_cache_key"]
        all_cache_value = inputs["all_cache_value"]

        # When `mask_mode` is `causal` , input_mask is not required
        # if self.mask_mode in ["user_defined"]:
        #     input_mask = inputs["input_mask"]

        if self.use_type_embeddings:
            input_type_ids = inputs["input_type_ids"]

        # sequence_length = tf.shape(input_ids)[1]

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
            positional_embeddings = self._position_embedding_layer(input_type_ids)
            embeddings = embeddings + positional_embeddings

        # Norm + dropout
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)

        # Initialize `attention_mask` as empty list
        attention_mask = []
        if self.mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        decoder_outputs = []
        position_bias = None
        decoder_encoder_position_bias = None
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            # Fetching
            cache_value = all_cache_value[i]
            cache_key = all_cache_key[i]

            (embeddings, position_bias, decoder_encoder_position_bias, cache_key, cache_value,) = layer(
                [
                    embeddings,
                    attention_mask,
                    encoder_hidden_state,
                    decoder_encoder_mask,
                ],
                position_bias=position_bias,
                decoder_encoder_position_bias=decoder_encoder_position_bias,
                cache_key=cache_key,
                cache_value=cache_value,
            )

            # Updating
            all_cache_key[i] = cache_key
            all_cache_value[i] = cache_value

            decoder_outputs.append(embeddings)

        # Stack all layers key and value together
        # num_layers x batch_size x num_heads x sequence_length x (hidden_dimension/num_heads) # noqa
        all_cache_key = tf.stack(all_cache_key, axis=0, name="all_cache_key")
        all_cache_value = tf.stack(all_cache_value, axis=0, name="all_cache_value")

        decoder_outputs[-1] = self._last_layer_norm(decoder_outputs[-1])
        # * (self.embedding_size ** -0.5)
        # batch_size x sequence_length x embedding_size
        token_embeddings = self._last_layer_dropout(decoder_outputs[-1])

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
            "token_logits": token_logits,
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
        # Decoder don't need this

        if self.mask_mode in ["user_defined"]:
            input_mask = inputs["input_mask"]

        if self.use_type_embeddings:
            input_type_ids = inputs["input_type_ids"]

        # sequence_length = tf.shape(input_ids)[1]

        # If decoder is not sharing embeddings
        word_embeddings = self._embedding_layer(input_ids)
        embeddings = word_embeddings
        # Add word_embeddings + position_embeddings + type_embeddings
        if self.use_type_embeddings:
            type_embeddings = self._type_embeddings(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self.use_positonal_embeddings:
            positional_embeddings = self._position_embedding_layer(input_type_ids)
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
        position_bias = None
        decoder_encoder_position_bias = None
        for i in range(self.num_hidden_layers):
            layer = self._transformer_layers[i]
            embeddings, position_bias, decoder_encoder_position_bias, _, _ = layer(
                [embeddings, attention_mask, encoder_output, decoder_encoder_mask],
                position_bias=position_bias,
                decoder_encoder_position_bias=decoder_encoder_position_bias,
            )
            decoder_outputs.append(embeddings)

        decoder_outputs[-1] = self._last_layer_norm(decoder_outputs[-1])
        # (self.embedding_size ** -0.5)
        # batch_size x sequence_length x embedding_size
        token_embeddings = self._last_layer_dropout(decoder_outputs[-1])

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

    def call(self, inputs):
        """Forward Pass.
        We have 2 pipelines . Training pipeline is relatively simpler
        Testing pipeline has few changes to
        accomodate caching of `key` and `value` for Transformer.
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
