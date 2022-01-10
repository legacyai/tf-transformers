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
"""ViTFeatureExtractor using Tensorflow Ops"""
from typing import Dict, Union

import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel

code_example = r'''

        >>> from tf_transformers.models import  ViTFeatureExtractorTF
        >>> image_path_list = # List fo image paths
        >>> vit_feature_extractor_tf = ViTFeatureExtractorTF(img_height=224, img_width=224)
        >>> outputs = vit_feature_extractor_tf({'image': tf.constant(image_path_list)})

'''


class ViTFeatureExtractorTF(LegacyLayer):
    def __init__(
        self,
        img_height,
        img_width,
        num_channels=3,
        rescale=True,
        normalize=True,
        scale_value=1.0 / 255.0,
        mean=0.5,
        variance=0.5,
        is_training=False,
        use_dropout=False,
        **kwargs,
    ):

        r"""
        VitFeatureExtractor using Tensorflow Ops, which allows complete serialization

            Args:
                img_height (:obj:`int`): Image Height to resize.
                img_width  (:obj:`int`): Image Width to resize.
                num_channels (:obj:`int`): Number of image channels.
                rescale (:obj:`bool`): To rescale the image. default (:obj:`True`).
                normalize (:obj:`bool`): To normalize the image. default (:obj:`True`).
                return_layer (:obj:`bool`): Whether to return tf.keras.layers.Layer/LegacyLayer.
                scale_value (:obj:`float`): Used for rescaling image. default (:obj:`1.0/255.0`).
                mean (:obj:`float`): Used for normalize image. default (:obj:`0.5`).
                variance (:obj:`float`): Used for normalize image. default (:obj:`0.5`).
            Returns:
                LegacyModel/LegacyLayer.

            Examples::

                {3}

        """
        super(ViTFeatureExtractorTF, self).__init__(
            is_training=is_training, use_dropout=use_dropout, name="vit_feature_extractor", **kwargs
        )

        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.rescale = rescale
        self.normalize = normalize
        self.scale_value = scale_value
        self.mean = mean
        self.variance = variance
        self.is_training = is_training
        self.use_dropout = use_dropout

        if self.rescale:
            self.rescaler = tf.keras.layers.Rescaling(scale=scale_value)
        if self.normalize:
            self.normalizer = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance)

        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def read_process_resize(self, image_path: str):
        """Read, decode and process"""
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=self.num_channels)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        return img

    def call(self, inputs: Dict[str, Union[tf.keras.layers.Input, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """Main call"""
        image_path = inputs['image']
        image_processed = tf.map_fn(
            self.read_process_resize, image_path, name='batch_processing', fn_output_signature=tf.float32
        )
        if self.rescale:
            image_processed = self.rescaler(image_processed)
        if self.normalize:
            image_processed = self.normalizer(image_processed)

        return {'input_pixels': image_processed}

    def get_config(self) -> Dict:
        """Return config"""
        config = {
            "img_height": self.img_height,
            "img_width": self.img_width,
            "num_channels": self.num_channels,
            "rescale": self.rescale,
            "normalize": self.normalize,
            "scale_value": self.scale_value,
            "mean": self.mean,
            "variance": self.variance,
            "is_training": self.is_training,
            "use_dropout": self.use_dropout,
        }
        base_config = super(ViTFeatureExtractorTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_model(self, initialize_only=False):
        inputs = {
            "image": tf.keras.layers.Input(
                shape=(),
                dtype=tf.string,
                name="image",
            )
        }
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name="vit_feature_extractor")
        model.model_config = self.get_config()
        return model
