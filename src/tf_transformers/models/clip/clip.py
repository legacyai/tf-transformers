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
"""The main wrapper to wrap Encoder Decoder models"""
import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import CALL_ENCODER_DOCSTRING_CLIP


class CLIPEncoder(LegacyLayer):
    def __init__(self, image_encoder, text_encoder, is_training, use_dropout, **kwargs):
        r"""
        CLIPEncoder

        Args:
            image_encoder (:obj:`tf_transformers.core.LegacyLayer`): Image Encoder
            text_encoder (:obj:`tf_transformers.core.LegacyLayer`): Text Encoder
            is_training (:obj:`bool`): For enabling training.
            use_dropout (:obj:`bool`): To enable dropout or not.
        Returns:
            LegacyModel/LegacyLayer .
        """
        super(CLIPEncoder, self).__init__(
            name='tftransformers/clip', is_training=is_training, use_dropout=use_dropout, **kwargs
        )

        assert isinstance(image_encoder, LegacyLayer)
        assert isinstance(text_encoder, LegacyLayer)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        assert self.image_encoder._config_dict['projection_dim'] == self.text_encoder._config_dict['projection_dim']

        self.logits_scale = tf.Variable(tf.math.log(1 / 0.07), name='logits_scale')
        # Initialize model
        self.model_inputs, self.model_ouputs = self.get_model(initialize_only=True)

    def get_model(self, initialize_only=False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: model (tf.keras.Layer) instance
            initialize_only: bool

        """

        image_inputs = self.image_encoder.model_inputs
        text_inputs = self.text_encoder.model_inputs

        inputs = {}

        for k, v in image_inputs.items():
            inputs[k] = v
        for k, v in text_inputs.items():
            inputs[k] = v

        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self.name)
        config = {}
        config['vision_config'] = self.image_encoder._config_dict
        config['text_config'] = self.text_encoder._config_dict
        model.model_config = config
        return model

    @add_start_docstrings(
        "Forward pass of CLIP :",
        CALL_ENCODER_DOCSTRING_CLIP,
    )
    def call(self, inputs):
        """Call"""

        # Image inputs
        image_inputs = {}
        if "input_pixels" in inputs:
            image_inputs["input_pixels"] = inputs["input_pixels"]
        # Text inputs
        text_inputs = {}
        for input_name in ["input_ids", "input_mask", "input_type_ids"]:
            if input_name in inputs:
                text_inputs[input_name] = inputs[input_name]

        image_outputs = self.image_encoder(image_inputs)
        text_outputs = self.text_encoder(text_inputs)

        # Image Projection
        image_features_unnormalized = image_outputs['cls_output']
        # Text Projection
        text_features_unnormalized = text_outputs['cls_output']

        # Normalize
        image_features = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(image_features_unnormalized)
        text_features = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(text_features_unnormalized)

        # cosine similarity as logits
        logits_scale = tf.clip_by_value(
            self.logits_scale, clip_value_min=tf.math.log(1 / 0.07), clip_value_max=4.6051752
        )

        logits_scale = tf.math.exp(logits_scale)

        logits_per_text = tf.matmul(text_features, image_features, transpose_b=True) * logits_scale
        logits_per_image = tf.matmul(image_features, text_features, transpose_b=True) * logits_scale

        outputs = {}
        for output_name, tensor in image_outputs.items():
            outputs['image_' + output_name] = tensor

        for output_name, tensor in text_outputs.items():
            outputs['text_' + output_name] = tensor

        outputs['image_features'] = image_features_unnormalized
        outputs['text_features'] = text_features_unnormalized
        outputs['image_features_normalized'] = image_features
        outputs['text_features_normalized'] = text_features
        outputs['logits_per_text'] = logits_per_text
        outputs['logits_per_image'] = logits_per_image

        # Iterate over all layers if required
        all_layer_logits_per_image = []
        all_layer_logits_per_text = []
        if 'all_layer_cls_output' in text_outputs and 'all_layer_cls_output' in image_outputs:
            num_layers = len(text_outputs['all_layer_cls_output'])
            for i in range(num_layers):
                # Image Projection
                image_features_unnormalized = image_outputs['all_layer_cls_output'][i]
                # Text Projection
                text_features_unnormalized = text_outputs['all_layer_cls_output'][i]

                # Normalize
                image_features = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                    image_features_unnormalized
                )
                text_features = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                    text_features_unnormalized
                )

                # cosine similarity as logits
                logits_scale = tf.clip_by_value(
                    self.logits_scale, clip_value_min=tf.math.log(1 / 0.07), clip_value_max=4.6051752
                )

                logits_scale = tf.math.exp(logits_scale)

                logits_per_text = tf.matmul(text_features, image_features, transpose_b=True) * logits_scale
                logits_per_image = tf.matmul(image_features, text_features, transpose_b=True) * logits_scale

                all_layer_logits_per_image.append(logits_per_image)
                all_layer_logits_per_text.append(logits_per_text)

            outputs['all_layer_logits_per_image'] = all_layer_logits_per_image
            outputs['all_layer_logits_per_text'] = all_layer_logits_per_text

        return outputs
