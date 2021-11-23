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
"""TF 2.0 ViT Model"""

from typing import Dict, Union

import tensorflow as tf
from absl import logging

from tf_transformers.activations import get_activation
from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.layers import PatchEmbeddings, PositionEmbeddingImage
from tf_transformers.layers.transformer import TransformerVIT
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    CALL_ENCODER_DOCSTRING,
    ENCODER_CLASS_DOCSTRING,
)

logging.set_verbosity("INFO")


@add_start_docstrings(
    "ViT Model :",
    ENCODER_CLASS_DOCSTRING.format("tf_transformers.models.vit.ViTConfig"),
)
class ViTEncoder(LegacyLayer):
    def __init__(
        self,
        config: Dict,
        mask_mode: str = "user_defined",
        name: str = "vit",
        use_dropout: bool = False,
        is_training: bool = False,
        use_auto_regressive: bool = False,
        use_decoder: bool = False,
        batch_size: bool = None,
        sequence_length: bool = None,
        return_all_layer_outputs: bool = False,
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
        self._return_all_layer_outputs = return_all_layer_outputs

        if "num_labels" not in config:
            config["num_labels"] = None
            self._num_labels = config['num_labels']

        self._patch_size = config['patch_size']
        self._image_size = config['image_size']
        self._num_channels = config['num_channels']

        one_side_patch = config['image_size'] // config['patch_size']
        self._num_patches = (one_side_patch * one_side_patch) + 1  # 1 for CLS token

        # self._self_setattr_tracking = False
        super(ViTEncoder, self).__init__(
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
            "return_all_layer_outputs": self._return_all_layer_outputs,
            "num_patches": self._num_patches,
            "patch_size": self._patch_size,
            "image_size": self._image_size,
            "num_channels": self._num_channels,
        }
        # Update config dict with passed config
        self._config_dict.update(config)

        self._cls_token = tf.Variable(
            tf.zeros((1, 1, config['embedding_size'])), name='{}/cls_token'.format(self._model_name)
        )
        self._embedding_layer = PatchEmbeddings(
            config['image_size'], config['patch_size'], config['num_channels'], config['embedding_size']
        )
        self._positional_embedding_layer = PositionEmbeddingImage(self._num_patches, config['embedding_size'])

        # Embedding Norm
        self._last_layer_norm = tf.keras.layers.LayerNormalization(
            name="last_layer_norm", axis=-1, epsilon=config["layer_norm_epsilon"], dtype=tf.float32
        )

        # Embedding dropout Layer
        self._embedding_dropout = tf.keras.layers.Dropout(rate=config["hidden_dropout_prob"])

        # Transformer Layer
        self._transformer_layers = []
        for i in range(config["num_hidden_layers"]):
            layer = TransformerVIT(
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

        # Add Classifier layer (by default VIT traines on 1000 labels)
        if config['num_labels']:

            self._classifier_layer = tf.keras.layers.Dense(
                units=config["num_labels"],
                activation=None,
                kernel_initializer=self._initializer,
                name="classifier_layer",
            )

        # CLS layer
        self._pooler_layer = tf.keras.layers.Dense(
            units=config["embedding_size"],
            activation="tanh",
            kernel_initializer=self._initializer,
            name="pooler_transform",
        )

        self.call_fn = self.get_call_method(self._config_dict)
        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def get_model(self, initialize_only=False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: model (tf.keras.Layer) instance
            initialize_only: If False, model (LegacyModel) wont be returned.

        """

        input_ids = tf.keras.layers.Input(
            shape=(self._config_dict['image_size'], self._config_dict['image_size'], self._config_dict['num_channels']),
            batch_size=self._batch_size,
            dtype=tf.float32,
            name="input_ids",
        )
        inputs = {}
        inputs["input_ids"] = input_ids  # Default

        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        # Adding model_config is a hack
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        model.model_config = self._config_dict
        return model

    @add_start_docstrings(
        "Forward pass of Vit :",
        CALL_ENCODER_DOCSTRING,
    )
    def call_encoder(self, inputs: Dict[str, Union[tf.keras.layers.Input, tf.Tensor]]) -> Dict[str, tf.Tensor]:

        # 1. Collect Patch Embeddings
        input_ids = inputs["input_ids"]
        batch_size = tf.shape(input_ids)[0]
        # b x one_side_patch x one_side_patch x embedding_size (b x 14 x 14 x 768)
        embeddings = self._embedding_layer(input_ids)
        # Reshape it to (b x (one_side_patch * one_side_patch) x embedding_size) (b x 196 x 768)
        embeddings = tf.reshape(embeddings, (batch_size, -1, self._config_dict['embedding_size']))
        # Add CLS token to the start (b x 197 x 768)
        # Replicate cls_vector (batch times) tf.tile dont work for some reason
        cls_token_tiled = tf.ones([batch_size, 1, 1]) * self._cls_token
        embeddings = tf.concat([cls_token_tiled, embeddings], axis=1)

        # Add word_embeddings + position_embeddings + type_embeddings
        #         if self._type_embeddings_layer:
        #             input_type_ids = inputs["input_type_ids"]
        #             type_embeddings = self._type_embeddings_layer(input_type_ids)
        #             embeddings = embeddings + type_embeddings

        # Addition happens internally
        if self._positional_embedding_layer:
            embeddings = self._positional_embedding_layer(embeddings)

        # 3. Attention  Mask
        attention_mask = tf.ones((batch_size, self._num_patches, self._num_patches))
        # 4. Transformer Outputs
        encoder_outputs = []
        for i in range(self._config_dict["num_hidden_layers"]):
            layer = self._transformer_layers[i]
            embeddings, _, _ = layer([embeddings, attention_mask])
            encoder_outputs.append(embeddings)

        # batch_size x sequence_length x embedding_size
        token_embeddings = self._last_layer_norm(encoder_outputs[-1])

        # First word of last layer outputs [CLS]
        cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(token_embeddings)
        # batch_size x embedding_size
        cls_output = self._pooler_layer(cls_token_tensor)

        result = {"token_embeddings": token_embeddings, "cls_output": cls_output, "cls_token_tensor": cls_token_tensor}
        if self._config_dict['num_labels']:
            classifier_predictions = self._classifier_layer(cls_token_tensor)
            result['classifier_predictions'] = classifier_predictions

        if self._return_all_layer_outputs:
            all_cls_token_tensors = []
            all_cls_output = []
            all_layer_classifier_predictions = []
            for per_layer_token_embeddings in encoder_outputs:
                per_layer_token_embeddings = self._last_layer_norm(per_layer_token_embeddings)
                per_cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                    per_layer_token_embeddings
                )
                all_cls_token_tensors.append(per_cls_token_tensor)
                all_cls_output.append(self._pooler_layer(per_cls_token_tensor))

                if self._config_dict['num_labels']:
                    classifier_predictions = self._classifier_layer(cls_token_tensor)
                    all_layer_classifier_predictions.append(classifier_predictions)

            result["all_layer_token_embeddings"] = encoder_outputs
            result["all_layer_cls_output"] = all_cls_output
            result["all_layer_cls_token_tensor"] = all_cls_token_tensors
            if self._config_dict['num_labels']:
                result["all_layer_classifier_predictions"] = all_layer_classifier_predictions

        return result

    def call_encoder_auto_regressive(self, inputs):
        raise NotImplementedError("ViT as of now not supports decoding")

    def call_decoder(self, inputs):
        raise NotImplementedError("As of now Vit doesn't support Decoder")

    def call_decoder_auto_regressive(self, inputs):
        raise NotImplementedError("As of now ViT doesn't support Seq2Seq decoding")

    def call(self, inputs):
        """Call method"""
        outputs = self.call_fn(inputs)
        return outputs

    def get_embedding_table(self):
        return NotImplementedError

    def get_config(self):
        return self._config_dict

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
