# coding=utf-8
# Copyright 2021 TF-Transformers Authors and The TensorFlow Authors.
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
"""Mix Language Model with 3d input mask"""
import tensorflow as tf

from tf_transformers.core import LegacyModel
from tf_transformers.models import RobertaEncoder
from tf_transformers.utils import tf_utils


class MixEncoder(RobertaEncoder):
    def __init__(self, config, **kwargs):
        super(MixEncoder, self).__init__(config, **kwargs)

    def get_model(self, initialize_only=False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: model (tf.keras.Layer) instance
        """

        input_ids = tf.keras.layers.Input(
            shape=(self._sequence_length,),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="input_ids",
        )
        input_mask = tf.keras.layers.Input(
            shape=(self._sequence_length, self._sequence_length),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="input_mask_3d",
        )
        input_type_ids = tf.keras.layers.Input(
            shape=(self._sequence_length,),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="input_type_ids",
        )

        inputs = {}
        inputs["input_ids"] = input_ids  # Default
        inputs["input_mask_3d"] = input_mask

        # If type mebddings required
        if self._type_embeddings_layer:
            inputs["input_type_ids"] = input_type_ids
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        # Adding model_config is a hack
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        model.model_config = self._config_dict
        return model

    def call_encoder(self, inputs):
        """Forward pass of an Encoder

        Args:
            inputs ([dict of tf.Tensor]): This is the input to the model.

            'input_ids'         --> tf.int32 (b x s)
            'input_mask'        --> tf.int32 (b x s) # optional
            'input_type_ids'    --> tf.int32 (b x s) # optional

        Returns:
            [dict of tf.Tensor]: Output from the model

            'cls_output'        --> tf.float32 (b x s) # optional
            'token_embeddings'  --> tf.float32 (b x s x h)
            'all_layer_token_embeddings' --> tf.float32 (List of (b x s x h)
                                            from all layers)
            'all_layer_cls_output'       --> tf.float32 (List of (b x s)
                                            from all layers)
        """

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
        attention_mask = inputs['input_mask_3d']

        # 4. Transformer Outputs
        encoder_outputs = []
        for i in range(self._config_dict["num_hidden_layers"]):
            layer = self._transformer_layers[i]
            embeddings, _, _ = layer([embeddings, attention_mask])
            encoder_outputs.append(embeddings)

        # First word of last layer outputs [CLS]
        cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(encoder_outputs[-1])
        # batch_size x embedding_size
        cls_output = self._pooler_layer(cls_token_tensor)
        # batch_size x sequence_length x embedding_size
        token_embeddings = encoder_outputs[-1]

        token_logits = tf.matmul(
            tf.cast(token_embeddings, dtype=tf_utils.get_dtype()),
            tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
            transpose_b=True,
        )

        last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

        result = {
            "cls_output": cls_output,
            "token_embeddings": token_embeddings,
            "token_logits": token_logits,
            "last_token_logits": last_token_logits,
        }

        if self._return_all_layer_outputs:
            all_cls_output = []
            all_token_logits = []
            for per_layer_token_embeddings in encoder_outputs:
                per_layer_token_embeddings = tf.cast(per_layer_token_embeddings, dtype=tf_utils.get_dtype())
                per_cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                    per_layer_token_embeddings
                )
                all_cls_output.append(self._pooler_layer(per_cls_token_tensor))

                layer_token_logits = tf.matmul(
                    per_layer_token_embeddings,
                    tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
                    transpose_b=True,
                )
                all_token_logits.append(layer_token_logits)

            result["all_layer_token_embeddings"] = encoder_outputs
            result["all_layer_cls_output"] = all_cls_output
            result["all_layer_token_logits"] = all_token_logits

        return result
