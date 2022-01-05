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
"""TF 2.0 MT5 Model"""

from typing import Dict, Union

import tensorflow as tf
from absl import logging

from tf_transformers.activations import get_activation
from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.layers import T5LayerNormalization
from tf_transformers.layers.mask import CausalMask, SelfAttentionMask, prefix_mask
from tf_transformers.layers.transformer import TransformermT5
from tf_transformers.utils import tf_utils
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    CALL_DECODER_AUTO_REGRESSIVE_DOCSTRING,
    CALL_DECODER_DOCSTRING,
    CALL_ENCODER_DOCSTRING,
    ENCODER_CLASS_DOCSTRING,
    MAIN_CALL_DOCSTRING,
)

logging.set_verbosity("INFO")


@add_start_docstrings(
    "MT5 Model :",
    ENCODER_CLASS_DOCSTRING.format("tf_transformers.models.mt5.MT5Config"),
)
class MT5Encoder(LegacyLayer):
    def __init__(
        self,
        config,
        mask_mode="user_defined",
        name="mt5",
        use_dropout=False,
        is_training=False,
        use_auto_regressive=False,
        use_decoder=False,
        batch_size=None,
        sequence_length=None,
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
        self._return_all_layer_outputs = return_all_layer_outputs

        # self._self_setattr_tracking = False
        super(MT5Encoder, self).__init__(
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
        # self._embedding_norm = GPT2LayerNormalization(
        #     name="embeddings/layer_norm", axis=-1, epsilon=config["layer_norm_epsilon"], dtype=tf.float32
        # )

        # Embedding dropout Layer
        self._embedding_dropout = tf.keras.layers.Dropout(rate=config["hidden_dropout_prob"])
        # Transformer Layer
        self._transformer_layers = []
        for i in range(config["num_hidden_layers"]):
            #  Required only for first layer to create the positonal_embeddings
            if i == 0:
                create_positonal_embedding = True
            else:
                create_positonal_embedding = False
            layer = TransformermT5(
                hidden_size=config["embedding_size"],
                num_attention_heads=config["num_attention_heads"],
                attention_head_size=config["attention_head_size"],
                intermediate_size=config["intermediate_size"],
                intermediate_activation=self._intermediate_activation,
                bidirectional=config["bidirectional"],
                create_positonal_embedding=create_positonal_embedding,
                positional_buckets=config["positional_buckets"],
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

        # Last Layer Normalization (only in GPT2)
        self._last_layer_norm = T5LayerNormalization(
            name="last_layer_norm",
            axis=-1,
            epsilon=config["layer_norm_epsilon"],
            dtype=tf.float32,
        )
        self._last_layer_dropout = tf.keras.layers.Dropout(rate=config["hidden_dropout_prob"])
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
        # masked_lm_positions = tf.keras.layers.Input(
        #     shape=(None,),
        #     batch_size=self._batch_size,
        #     dtype=tf.int32,
        #     name="masked_lm_positions",
        # )
        inputs = {}
        inputs["input_ids"] = input_ids  # Default
        # if mask_mode != 'causal', user has to provde mask
        if self._mask_mode != "causal":
            inputs["input_mask"] = input_mask
        # If type mebddings required
        if self._type_embeddings_layer:
            inputs["input_type_ids"] = input_type_ids
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
        "Forward pass of MT5 :",
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
        # embeddings = self._embedding_norm(embeddings)
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
        position_bias = None
        for i in range(self._config_dict["num_hidden_layers"]):
            layer = self._transformer_layers[i]
            embeddings, position_bias, k, v = layer([embeddings, attention_mask], position_bias=position_bias)
            encoder_outputs.append(embeddings)

        token_embeddings = self._last_layer_norm(encoder_outputs[-1])
        token_embeddings = self._last_layer_dropout(token_embeddings)
        # First word of last layer outputs [CLS]
        # batch_size x embedding_size
        # batch_size x sequence_length x embedding_size
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
            for per_layer_token_embeddings in encoder_outputs:
                per_layer_token_embeddings = self._last_layer_norm(per_layer_token_embeddings)
                per_layer_token_embeddings = self._last_layer_dropout(per_layer_token_embeddings)
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
        "MT5 Encoder Text Generation :",
        "MT5 encoder standalone text generation is not yet implemnted due to relative postional embedding",
    )
    def call_encoder_auto_regressive(
        self, inputs: Dict[str, Union[tf.keras.layers.Input, tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        raise NotImplementedError()

    @add_start_docstrings(
        "Forward pass of MT5 Decoder :",
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
        # embeddings = self._embedding_norm(embeddings)
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
        position_bias = None
        decoder_encoder_position_bias = None
        for i in range(self._config_dict["num_hidden_layers"]):
            layer = self._transformer_layers[i]
            embeddings, position_bias, decoder_encoder_position_bias, _, _ = layer(
                [embeddings, attention_mask, encoder_output, decoder_encoder_mask],
                position_bias=position_bias,
                decoder_encoder_position_bias=decoder_encoder_position_bias,
            )
            decoder_outputs.append(embeddings)

        token_embeddings = self._last_layer_norm(decoder_outputs[-1])
        # batch_size x sequence_length x embedding_size
        token_embeddings = self._last_layer_dropout(token_embeddings)

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
                per_layer_token_embeddings = self._last_layer_norm(per_layer_token_embeddings)
                per_layer_token_embeddings = self._last_layer_dropout(per_layer_token_embeddings)
                layer_token_logits = tf.matmul(
                    per_layer_token_embeddings,
                    tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
                    transpose_b=True,
                )
                all_token_logits.append(layer_token_logits)

            result["all_layer_token_embeddings"] = decoder_outputs
            result["all_layer_token_logits"] = all_token_logits

        return result

    @add_start_docstrings(
        "Forward pass of MT5 Decoder Auto Regressive/ Text Generation :",
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

        all_cache_key = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_key, num_or_size_splits=self._config_dict["num_hidden_layers"], axis=0)
        ]
        all_cache_value = [
            tf.squeeze(item, axis=0)
            for item in tf.split(all_cache_value, num_or_size_splits=self._config_dict["num_hidden_layers"], axis=0)
        ]

        # 1. Collect Word Embeddings
        embeddings = self._embedding_layer(input_ids)
        # Add word_embeddings + position_embeddings + type_embeddings
        if self._type_embeddings_layer:
            input_type_ids = inputs["input_type_ids"]
            type_embeddings = self._type_embeddings_layer(input_type_ids)
            embeddings = embeddings + type_embeddings
        if self._positional_embedding_layer:

            # T5/MT5 doesn't have positional embedding, so this is not useful
            # Still, in case to use it in future, check def call_decoder_auto_regressive in Roberta Model
            sequence_length = -1  # Dummy Assignment to avoid mypy error.
            positional_embeddings = self._positional_embedding_layer(sequence_length)  # noqa
            # Make it 3D for sum ( For decoder we decode one at a time)
            positional_embeddings = tf.expand_dims(positional_embeddings, 0)
            embeddings = embeddings + positional_embeddings

        # Norm + dropout
        # embeddings = self._embedding_norm(embeddings)
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
        position_bias = None
        decoder_encoder_position_bias = None
        for i in range(self._config_dict["num_hidden_layers"]):
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

        decoder_outputs[-1] = self._last_layer_norm(decoder_outputs[-1])
        # Stack all layers key and value together
        # num_layers x batch_size x num_heads x sequence_length x (hidden_dimension/num_heads)
        all_cache_key = tf.stack(all_cache_key, axis=0, name="all_cache_key")
        all_cache_value = tf.stack(all_cache_value, axis=0, name="all_cache_value")

        # batch_size x sequence_length x embedding_size
        token_embeddings = self._last_layer_dropout(decoder_outputs[-1])
        token_logits = tf.matmul(token_embeddings, self.get_embedding_table(), transpose_b=True)
        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        return {
            "token_embeddings": token_embeddings,
            "all_cache_key": all_cache_key,
            "all_cache_value": all_cache_value,
            "last_token_logits": last_token_logits,
        }

    @add_start_docstrings(
        "MT5 Call method :",
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
