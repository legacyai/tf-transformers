# coding=utf-8
# Copyright 2021 TF-Transformers Authors.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TF 2.0 Bart Model"""

from typing import Dict, Union

import tensorflow as tf
from absl import logging

from tf_transformers.activations import get_activation
from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.layers.mask import CausalMask, SelfAttentionMask, prefix_mask
from tf_transformers.layers.transformer import TransformerBART
from tf_transformers.utils import tf_utils
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    CALL_DECODER_AUTO_REGRESSIVE_DOCSTRING,
    CALL_DECODER_DOCSTRING,
    CALL_ENCODER_AUTO_REGRESSIVE_DOCSTRING,
    CALL_ENCODER_DOCSTRING,
    ENCODER_CLASS_DOCSTRING,
    MAIN_CALL_DOCSTRING,
)

logging.set_verbosity("INFO")

BART_DECODER_START_TOKEN_ID = 2


@add_start_docstrings(
    "Bart Model :",
    ENCODER_CLASS_DOCSTRING.format("tf_transformers.models.bart.BartConfig"),
)
class BartEncoder(LegacyLayer):
    def __init__(
        self,
        config,
        mask_mode="user_defined",
        name="bart",
        use_dropout=False,
        is_training=False,
        use_auto_regressive=False,
        use_decoder=False,
        batch_size=None,
        sequence_length=None,
        use_mlm_layer=False,
        use_masked_lm_positions=False,
        return_all_layer_outputs=False,
        **kwargs,
    ):

        # IMPORTANT: Because saved_model causes some serialization problems here
        # self.config              = config

        # Default initializer
        _stddev = config["initializer_range"]
        self._initializer = tf.keras.initializers.TruncatedNormal(stddev=_stddev)
        self._initializer = tf.keras.initializers.get(self._initializer)
        self._activation = get_activation(config["hidden_act"])
        self._intermediate_activation = get_activation(config["intermediate_act"])

        self._mask_mode = mask_mode
        self._model_name = "tf_transformers/" + name
        self._use_dropout = use_dropout
        self._is_training = is_training
        self._use_auto_regressive = use_auto_regressive
        self._use_decoder = use_decoder
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._use_masked_lm_positions = use_masked_lm_positions
        self._use_mlm_layer = use_mlm_layer
        self._return_all_layer_outputs = return_all_layer_outputs

        # self._self_setattr_tracking = False
        super(BartEncoder, self).__init__(
            is_training=self._is_training, use_dropout=self._use_dropout, name=self._model_name, **kwargs
        )

        # Configuration
        self._config_dict = {
            "initializer": tf.keras.initializers.serialize(self._initializer),
            "activation": tf.keras.activations.serialize(self._activation),
            "mask_mode": self._mask_mode,
            "name": self._model_name,
            "is_training": self._is_training,
            "use_auto_regressive": self._use_auto_regressive,
            "use_decoder": self._use_decoder,
            "use_dropout": self._use_dropout,
            "batch_size": self._batch_size,
            "sequence_length": self._sequence_length,
            "use_mlm_layer": self._use_mlm_layer,
            "use_masked_lm_positions": self._use_masked_lm_positions,
            "return_all_layer_outputs": self._return_all_layer_outputs,
        }
        # Update config dict with passed config
        self._config_dict.update(config)

        # Call embedding layers
        (
            self._embedding_layer,
            self._type_embeddings_layer,
            self._positional_embedding_layer,
        ) = self.get_embedding_layers(self._config_dict)

        # Embedding Norm
        self._embedding_norm = tf.keras.layers.LayerNormalization(
            name="embeddings/layer_norm", axis=-1, epsilon=config["layer_norm_epsilon"], dtype=tf.float32
        )

        # Embedding dropout Layer
        self._embedding_dropout = tf.keras.layers.Dropout(rate=config["hidden_dropout_prob"])

        # Transformer Layer
        self._transformer_layers = []
        for i in range(config["num_hidden_layers"]):
            layer = TransformerBART(
                hidden_size=config["embedding_size"],
                num_attention_heads=config["num_attention_heads"],
                attention_head_size=config["attention_head_size"],
                intermediate_size=config["intermediate_size"],
                intermediate_activation=self._intermediate_activation,
                dropout_rate=config["hidden_dropout_prob"],
                attention_dropout_rate=config["attention_probs_dropout_prob"],
                kernel_initializer=self._initializer,
                is_training=self._is_training,
                use_dropout=self._use_dropout,
                use_decoder=self._use_decoder,
                layer_norm_epsilon=config["layer_norm_epsilon"],
                use_auto_regressive=self._use_auto_regressive,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        # # CLS layer
        # self._pooler_layer = tf.keras.layers.Dense(
        #     units=config["embedding_size"],
        #     activation="tanh",
        #     kernel_initializer=self._initializer,
        #     name="pooler_transform",
        # )

        self.call_fn = self.get_call_method(self._config_dict)
        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def get_model(self: LegacyLayer, initialize_only: bool = False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: Model layer
            initialize_only: If False, model (LegacyModel) wont be returned.
        """

        input_ids = tf.keras.layers.Input(
            shape=(self._sequence_length,),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="input_ids",
        )
        input_mask = tf.keras.layers.Input(
            shape=(self._sequence_length,),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="input_mask",
        )
        input_type_ids = tf.keras.layers.Input(
            shape=(self._sequence_length,),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="input_type_ids",
        )
        masked_lm_positions = tf.keras.layers.Input(
            shape=(None,),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="masked_lm_positions",
        )
        inputs = {}
        inputs["input_ids"] = input_ids  # Default
        # if mask_mode != 'causal', user has to provde mask
        if self._mask_mode != "causal":
            inputs["input_mask"] = input_mask
        # If type mebddings required
        if self._type_embeddings_layer:
            inputs["input_type_ids"] = input_type_ids
        # if masked_lm_positions
        if self._use_masked_lm_positions:
            inputs["masked_lm_positions"] = masked_lm_positions

        # Auto Regressive is activated only when is_training=False
        if self._is_training is False and self._use_auto_regressive:
            all_cache_key = tf.keras.layers.Input(
                shape=(
                    None,
                    self._config_dict["num_attention_heads"],
                    None,
                    self._config_dict["attention_head_size"],
                ),
                dtype=tf.float32,
                name="all_cache_key",
            )
            all_cache_value = tf.keras.layers.Input(
                shape=(
                    None,
                    self._config_dict["num_attention_heads"],
                    None,
                    self._config_dict["attention_head_size"],
                ),
                dtype=tf.float32,
                name="all_cache_value",
            )
            # Here batch_size = 1 , means we are dealing with vector for past_length
            past_length = tf.keras.layers.Input(shape=(None,), batch_size=1, dtype=tf.int32, name="past_length")
            inputs["all_cache_key"] = all_cache_key
            inputs["all_cache_value"] = all_cache_value
            inputs["past_length"] = past_length

        if self._use_decoder:
            # Encoder and Decoder shouldn't have same input name
            # when creating models
            for input_name in ["input_ids", "input_mask", "input_type_ids"]:
                if input_name in inputs:
                    inputs[input_name] = tf.keras.layers.Input(
                        shape=(self._sequence_length,),
                        batch_size=self._batch_size,
                        dtype=tf.int32,
                        name="decoder_{}".format(input_name),
                    )
            encoder_hidden_states = tf.keras.layers.Input(
                shape=(self._sequence_length, self._config_dict["embedding_size"]),
                batch_size=self._batch_size,
                dtype=tf.float32,
                name="encoder_hidden_states",
            )
            # batch_size x decoder_input_length x encoder_input_length
            decoder_encoder_mask = tf.keras.layers.Input(
                shape=(self._sequence_length, None),
                batch_size=self._batch_size,
                dtype=tf.float32,
                name="decoder_encoder_mask",
            )

            inputs["encoder_hidden_states"] = encoder_hidden_states
            inputs["decoder_encoder_mask"] = decoder_encoder_mask

            if "past_length" in inputs:
                del inputs["past_length"]

        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        # Adding model_config is a hack
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        model.model_config = self._config_dict
        return model

    @add_start_docstrings(
        "Forward pass of Bart :",
        CALL_ENCODER_DOCSTRING,
    )
    def call_encoder(self, inputs: Dict[str, Union[tf.keras.layers.Input, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        # 1. Collect Word Embeddings
        input_ids = inputs["input_ids"]
        sequence_length = tf.shape(input_ids)[1]
        embeddings = self._embedding_layer(input_ids)
        # Add word_embeddings + position_embeddings + type_embeddings
        if self._type_embeddings_layer:
            input_type_ids = inputs["input_type_ids"]
            type_embeddings = self._type_embeddings_layer(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self._positional_embedding_layer:
            positional_embeddings = self._positional_embedding_layer(tf.range(sequence_length))
            embeddings = embeddings + positional_embeddings

        # 2. Norm + dropout
        embeddings = self._embedding_norm(embeddings)
        embeddings = self._embedding_dropout(embeddings, training=self._use_dropout)

        # 3. Attention  Mask
        attention_mask = []
        if self._mask_mode == "user_defined":
            input_mask = inputs["input_mask"]
            attention_mask = SelfAttentionMask()([embeddings, input_mask])
        if self._mask_mode == "prefix":
            input_mask = inputs["input_mask"]
            attention_mask = tf.map_fn(prefix_mask, input_mask, dtype=tf.float32)
        if self._mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        # 4. Transformer Outputs
        encoder_outputs = []
        for i in range(self._config_dict["num_hidden_layers"]):
            layer = self._transformer_layers[i]
            embeddings, _, _ = layer([embeddings, attention_mask])
            encoder_outputs.append(embeddings)

        # First word of last layer outputs [CLS]
        # cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(encoder_outputs[-1])
        # batch_size x embedding_size
        # cls_output = self._pooler_layer(cls_token_tensor)
        # batch_size x sequence_length x embedding_size
        token_embeddings = encoder_outputs[-1]

        # Token logits
        token_logits = tf.matmul(
            tf.cast(token_embeddings, dtype=tf_utils.get_dtype()),
            tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
            transpose_b=True,
        )
        result = {"token_embeddings": token_embeddings, "token_logits": token_logits}
        if self._return_all_layer_outputs:
            all_token_logits = []
            for per_layer_token_embeddings in encoder_outputs:
                layer_token_logits = tf.matmul(
                    per_layer_token_embeddings,
                    tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
                    transpose_b=True,
                )
                all_token_logits.append(layer_token_logits)

            result["all_layer_token_embeddings"] = encoder_outputs
            result["all_layer_token_logits"] = all_token_logits

        return result

    @add_start_docstrings(
        "Forward pass of Bart Auto Regressive/ Text Generation :",
        CALL_ENCODER_AUTO_REGRESSIVE_DOCSTRING,
    )
    def call_encoder_auto_regressive(
        self, inputs: Dict[str, Union[tf.keras.layers.Input, tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:

        # 1. Gather necessary inputs
        input_ids_mod = inputs["input_ids"]
        all_cache_key = inputs["all_cache_key"]
        all_cache_value = inputs["all_cache_value"]
        past_length = inputs["past_length"]
        # Convert past_length 2D to 1D
        past_length = tf.squeeze(past_length, 0)

        # IMPORTANT : Get input_ids by replacing -1 with 0
        # Otherwise we get index error from word embeddings
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
        tf.assert_equal(tf.shape(all_cache_value)[0], self._config_dict["num_hidden_layers"])

        # Step 0 of inference. For step0, we do not have valid cache. We pass zero tensor
        def step_0(input_ids):
            sequence_length = tf.shape(input_ids)[1]
            positional_embeddings = self._positional_embedding_layer(tf.range(sequence_length))
            return sequence_length, positional_embeddings

        # From step_1 (autoregressive mode starts) onwards, we need to account for
        # `past_length` of previous words (inputs + generated) . Due to our logic,
        # we need to take a transpose of `position_embeddings` in this specific setting
        def step_other(input_ids):
            sequence_length = tf.shape(input_ids)[1]
            # Because past_length varies with batch
            positional_embeddings = self._positional_embedding_layer(past_length + sequence_length)
            positional_embeddings = tf.transpose(positional_embeddings, [1, 0, 2])
            return sequence_length, positional_embeddings

        # Split cache_key and cache_value into list (length = num_hudden_layers)
        # So, that each layer consumes the corresponding cache
        all_cache_key = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_key, num_or_size_splits=self._config_dict["num_hidden_layers"], axis=0)
        ]
        all_cache_value = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_value, num_or_size_splits=self._config_dict["num_hidden_layers"], axis=0)
        ]

        # Compuatation starts here
        # 1. Collect Word Embeddings
        embeddings = self._embedding_layer(input_ids)
        # Add word_embeddings + position_embeddings + type_embeddings
        if self._type_embeddings_layer:
            input_type_ids = inputs["input_type_ids"]
            type_embeddings = self._type_embeddings_layer(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self._positional_embedding_layer:
            # Condition to switch functions
            # if `sum(past_length) = 0` , means no outputs has been generated. First step (step_0)
            # the given inputs is the first input
            sequence_length, positional_embeddings = tf.cond(
                tf.equal(tf.reduce_sum(past_length), 0),
                lambda: step_0(input_ids),
                lambda: step_other(input_ids),
            )
            embeddings = embeddings + positional_embeddings

        # 2. Norm + dropout
        embeddings = self._embedding_norm(embeddings)
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)

        # 3. Attention Mask
        attention_mask = []
        if self._mask_mode == "user_defined":
            input_mask = inputs["input_mask"]
            attention_mask = SelfAttentionMask()([embeddings, input_mask])
        if self._mask_mode == "prefix":
            input_mask = inputs["input_mask"]
            attention_mask = tf.map_fn(prefix_mask, input_mask, fn_output_signature=tf.float32)
        if self._mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        # 4. Transformer Outputs
        encoder_outputs = []
        # Make all -1 positions to 0 (as -1 represents padding in the input)
        mask_values = tf.cast(tf.not_equal(input_ids_mod, -1), tf.float32)
        # We want zero values , where embeddings inputs where 0 (by replacing PAD -1)
        # So we use the mask and multiply it with embeddings
        embeddings = embeddings * tf.expand_dims(mask_values, -1)
        for i in range(self._config_dict["num_hidden_layers"]):
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

        # batch_size x sequence_length x embedding_size
        token_embeddings = encoder_outputs[-1]
        # token_logits = tf.matmul(token_embeddings, self.get_embedding_table(), transpose_b=True)

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
        # Add bias
        # Expand dims of past_length back to 2D
        past_length = tf.expand_dims(past_length, 0, name="past_length")
        # Stack all layers key and value together
        # num_layers x batch_size x num_heads x sequence_length x (hidden_dimension/num_heads)
        all_cache_key = tf.stack(all_cache_key, axis=0, name="all_cache_key")
        all_cache_value = tf.stack(all_cache_value, axis=0, name="all_cache_value")

        return {
            "token_embeddings": token_embeddings,
            "past_length": past_length,
            "all_cache_key": all_cache_key,
            "all_cache_value": all_cache_value,
            "last_token_logits": last_token_logits,
        }

    @add_start_docstrings(
        "Forward pass of Bart Decoder :",
        CALL_DECODER_DOCSTRING,
    )
    def call_decoder(self, inputs: Dict[str, Union[tf.keras.layers.Input, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        input_ids = inputs["input_ids"]
        encoder_output = inputs["encoder_hidden_states"]
        decoder_encoder_mask = inputs["decoder_encoder_mask"]

        # 1. Collect Word Embeddings
        sequence_length = tf.shape(input_ids)[1]
        embeddings = self._embedding_layer(input_ids)
        # Add word_embeddings + position_embeddings + type_embeddings
        if self._type_embeddings_layer:
            input_type_ids = inputs["input_type_ids"]
            type_embeddings = self._type_embeddings_layer(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self._positional_embedding_layer:
            positional_embeddings = self._positional_embedding_layer(tf.range(sequence_length))
            embeddings = embeddings + positional_embeddings

        # 2. Norm + dropout
        embeddings = self._embedding_norm(embeddings)
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)

        # 3. Attention  Mask
        attention_mask = []
        if self._mask_mode == "user_defined":
            input_mask = inputs["input_mask"]
            attention_mask = SelfAttentionMask()([embeddings, input_mask])
        if self._mask_mode == "prefix":
            input_mask = inputs["input_mask"]
            attention_mask = tf.map_fn(prefix_mask, input_mask, dtype=tf.float32)
        if self._mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)

        # Trasformer Outputs
        decoder_outputs = []
        for i in range(self._config_dict["num_hidden_layers"]):
            layer = self._transformer_layers[i]
            embeddings, _key, _value = layer([embeddings, attention_mask, encoder_output, decoder_encoder_mask])
            decoder_outputs.append(embeddings)

        token_embeddings = decoder_outputs[-1]
        token_logits = tf.matmul(
            token_embeddings, tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()), transpose_b=True
        )
        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        result = {
            "token_embeddings": token_embeddings,
            "token_logits": token_logits,
            "last_token_logits": last_token_logits,
        }

        if self._return_all_layer_outputs:
            all_token_logits = []
            for per_layer_token_embeddings in decoder_outputs:
                layer_token_logits = tf.matmul(
                    per_layer_token_embeddings,
                    tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
                    transpose_b=True,
                )
                # Add bias
                all_token_logits.append(layer_token_logits)

            result["all_layer_token_embeddings"] = decoder_outputs
            result["all_layer_token_logits"] = all_token_logits

        return result

    @add_start_docstrings(
        "Forward pass of Bart Decoder Auto Regressive/ Text Generation :",
        CALL_DECODER_AUTO_REGRESSIVE_DOCSTRING,
    )
    def call_decoder_auto_regressive(
        self, inputs: Dict[str, Union[tf.keras.layers.Input, tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:

        input_ids = inputs["input_ids"]
        encoder_hidden_state = inputs["encoder_hidden_states"]
        decoder_encoder_mask = inputs["decoder_encoder_mask"]
        all_cache_key = inputs["all_cache_key"]
        all_cache_value = inputs["all_cache_value"]

        #  We pass zero tensor for cache_key/cache_value at step 0, with decoder_sequence_length =1
        # Assuming we start by considering "one" decoder token to condition on
        # Even for step 1 , decoder_sequence_length = 1 remains same , as we start concacatanating
        # cache_key/cache_value from step 2 onwards. So, for step 0, decoder_sequence_length =
        # tf.shape(all_cache_key)[3] -1
        def _get_decoder_sequence_length_step0(input_ids):
            decoder_current_sequence_length = tf.shape(all_cache_key)[3] - 1
            return decoder_current_sequence_length

        # From step 1, we do not substract - 1, we just use it as decoder_sequence_length is aligned
        # from step 1
        def _get_decoder_sequence_length_step_other(input_ids):
            decoder_current_sequence_length = tf.shape(all_cache_key)[3]
            return decoder_current_sequence_length

        # This is useful only for positional embedding layer. T5 models dont have this
        decoder_current_sequence_length = tf.cond(
            tf.equal(tf.reduce_sum(all_cache_key), 0),
            lambda: _get_decoder_sequence_length_step0(all_cache_key),
            lambda: _get_decoder_sequence_length_step_other(all_cache_key),
        )

        all_cache_key = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_key, num_or_size_splits=self._config_dict["num_hidden_layers"], axis=0)
        ]
        all_cache_value = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_value, num_or_size_splits=self._config_dict["num_hidden_layers"], axis=0)
        ]

        # 1. Collect Word Embeddings
        sequence_length = tf.shape(input_ids)[1]  # noqa
        embeddings = self._embedding_layer(input_ids)
        # Add word_embeddings + position_embeddings + type_embeddings
        if self._type_embeddings_layer:
            input_type_ids = inputs["input_type_ids"]
            type_embeddings = self._type_embeddings_layer(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self._positional_embedding_layer:

            positional_embeddings = self._positional_embedding_layer(decoder_current_sequence_length)
            # Make it 3D for sum ( For decoder we decode one at a time)
            positional_embeddings = tf.expand_dims(positional_embeddings, 0)
            embeddings = embeddings + positional_embeddings

        # Norm + dropout
        embeddings = self._embedding_norm(embeddings)
        embeddings = self._embedding_dropout(embeddings, training=self.use_dropout)
        attention_mask = []
        if self._mask_mode == "causal":
            attention_mask = CausalMask()(embeddings)
        else:
            raise ValueError(
                "In Decoder mode. \
                Auto regressive always expect 'causal' mask mode But {} is provided".format(
                    self._mask_mode
                )
            )

        decoder_outputs = []
        for i in range(self._config_dict["num_hidden_layers"]):
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
        token_logits = tf.matmul(token_embeddings, self.get_embedding_table(), transpose_b=True)
        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        return {
            "token_embeddings": token_embeddings,
            "all_cache_key": all_cache_key,
            "all_cache_value": all_cache_value,
            "last_token_logits": last_token_logits,
        }

    @add_start_docstrings(
        "Bart Call method :",
        MAIN_CALL_DOCSTRING,
    )
    def call(self, inputs: Dict[str, tf.Tensor]):
        outputs = self.call_fn(inputs)
        return outputs

    def get_embedding_table(self) -> tf.Tensor:
        """Return Embedding Table"""
        return self._embedding_layer.embeddings

    def get_config(self) -> Dict:
        """Return Configuration dict"""
        return self._config_dict

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
