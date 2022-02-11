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

"""Keras-based patch embedding layer."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Image")
class PatchEmbeddings(tf.keras.layers.Layer):
    """Creates a patch embedding.

    This layer creates a patch embedding as described in "ViT":
    # TODO: Add paper link
    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_channels,
        embedding_width,
        initializer="glorot_uniform",
        dtype=tf.float32,
        use_bias=True,
        name="patch_embeddings",
        **kwargs,
    ):
        """
        Args:
            image_size (int): Height and width of image (Square Image)
            patch_size (int): Number of patches per block. ( 224 // 16 == 14 blocks)
            num_channels (int): Number of channels required (3 for RGB and 1 for GrayScale)
            embedding_width (int). Embedding dimension
            name (str, optional): name of the layer. Defaults to "patch_embeddings".
            dtype (tf.dtype, optional): [description]. Defaults to tf.float32.
        """
        super(PatchEmbeddings, self).__init__(name=name, dtype=dtype, **kwargs)
        self._image_size = image_size
        self._patch_size = patch_size
        self._num_channels = patch_size
        self._embedding_width = embedding_width
        self._use_bias = use_bias

    def get_config(self):
        """Config based on init arguments

        Returns:
            [dict]: Dict of all init arguments
        """
        config = {
            "image_size": self._image_size,
            "patch_size": self._patch_size,
            "num_channels": self._num_channels,
            "embedding_width": self._embedding_width,
            "name": self._name,
            "dtype": self._dtype,
        }
        base_config = super(PatchEmbeddings, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Build embeddings on run time (once)

        Args:
            input_shape ([TensorShape or List of TensorShape]): Shape of inputs
        """
        # self.conv = self.add_weight(
        #     "embeddings",
        #     shape=[self._max_sequence_length, self._embedding_width],
        #     initializer=self._initializer,
        # )
        self.conv = tf.keras.layers.Conv2D(
            self._embedding_width,
            kernel_size=(self._patch_size, self._patch_size),
            strides=(self._patch_size, self._patch_size),
            use_bias=self._use_bias,
        )

        super(PatchEmbeddings, self).build(input_shape)

    def call(self, inputs):
        """Call

        Args:
            inputs ([tf.Tensor]): input ids 4D (batch_size x img_height x img_width x num_channels)

        Returns:
            If image is (batch_size , 224 x 224 x 3), we return (batch_size x 196 x 768),
            if patch_size = 16
            [tf.Tensor]: embeddings 3D (b x (n_blocks*n_blocks) x h)
        """
        patch_embeddings = self.conv(inputs)
        return patch_embeddings


@tf.keras.utils.register_keras_serializable(package="Image")
class PositionEmbeddingImage(tf.keras.layers.Layer):
    """Creates a positional embedding as in Vit

    TODO Add details
    """

    def __init__(
        self,
        num_patches,
        embedding_width,
        initializer="glorot_uniform",
        name="positional_embeddings",
        dtype=tf.float32,
        **kwargs,
    ):
        """
        Args:
            num_patches (int): Num patches of image
            embedding_width (int): EMbedding dimension
        """
        super(PositionEmbeddingImage, self).__init__(name=name, dtype=dtype, **kwargs)
        self._num_patches = num_patches
        self._embedding_width = embedding_width
        self._initializer = initializer

    def get_config(self):
        """Config based on init arguments

        Returns:
            [dict]: Dict of all init arguments
        """
        config = {
            "num_patches": self._num_patches,
            "embedding_width": self._embedding_width,
            "initializer": self._initializer,
            "name": self._name,
            "dtype": self._dtype,
        }
        base_config = super(PositionEmbeddingImage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Build embeddings on run time (once)

        Args:
            input_shape ([TensorShape or List of TensorShape]): Shape of inputs
        """
        self.embeddings = self.add_weight(
            "embeddings",
            shape=[1, self._num_patches, self._embedding_width],
            initializer=self._initializer,
        )

        super(PositionEmbeddingImage, self).build(input_shape)

    def call(self, inputs):
        """Call

        Args:
            inputs ([tf.Tensor]): input ids 1D (tf.range(sequence_length))

        Returns:
            [tf.Tensor]: embeddings 3D (b x s x h)
        """
        # Add inputs to positional embeddings
        outputs = self.embeddings + inputs
        return outputs
