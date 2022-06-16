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
"""TF 2.0 UNET Model with Attention"""

import tensorflow as tf
import tensorflow_addons as tfa
from attention import ImageSelfAttention, ImageTextCrossAttention
from resnet import ResNetBlock

from tf_transformers.core import LegacyLayer, LegacyModel


class DownBlock(tf.keras.layers.Layer):
    """Add Residual block + Attention"""

    def __init__(
        self,
        out_channels,
        use_self_attention=False,
        use_cross_attention=False,
        use_scale_shift_norm=True,
        attention_emb=None,
        num_res_blocks=1,
        use_downsample=False,
        name='down_block',
        dtype=tf.float32,
        **kwargs,
    ):
        """
        Args:
        out_channels (int): Number of channels to project the images to

        """
        super(DownBlock, self).__init__(name=name, dtype=dtype, **kwargs)

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention

        self.downsample_block = tf.identity
        self.resnet_blocks = []
        for resnet_counter in range(num_res_blocks):
            self.resnet_blocks.append(
                ResNetBlock(
                    out_channels, use_scale_shift_norm=use_scale_shift_norm, name='resnet_{}'.format(resnet_counter)
                )
            )

        if use_downsample:
            self.downsample_block = tf.keras.layers.Conv2D(
                out_channels, kernel_size=(3, 3), strides=(2, 2), use_bias=True, padding='SAME', name='downsample'
            )
        if use_self_attention:
            self.self_attn_block = ImageSelfAttention()
        if use_cross_attention:
            self.cross_attention = ImageTextCrossAttention()

    def call(self, inputs):
        """Call

        [image, cemb, text_token_embeddings, text_input_mask]
        """
        image, cemb, text_token_embeddings, text_input_mask = inputs
        h = image
        for resnet_block in self.resnet_blocks:
            h = resnet_block([h, cemb])
        h = self.downsample_block(h)

        if self.use_self_attention:
            h, _, _ = self.self_attn_block([h, h])  # Image to Image
        if self.use_cross_attention:
            h, _, _ = self.cross_attention([h, text_token_embeddings, text_input_mask])

        return h


class MiddleBlock(tf.keras.layers.Layer):
    """Add Residual block + Attention"""

    def __init__(
        self,
        out_channels,
        use_scale_shift_norm=True,
        name='middle_block',
        dtype=tf.float32,
        **kwargs,
    ):
        """
        Args:
        out_channels (int): Number of channels to project the images to

        """
        super(MiddleBlock, self).__init__(name=name, dtype=dtype, **kwargs)

        self.resnet1 = ResNetBlock(out_channels, use_scale_shift_norm=use_scale_shift_norm, name='resnet_0')
        self.self_attention = ImageSelfAttention()
        self.resnet2 = ResNetBlock(out_channels, use_scale_shift_norm=use_scale_shift_norm, name='resnet_0')

    def call(self, inputs):
        """Call

        [image, cemb]
        """
        image, cemb = inputs
        h = self.resnet1([image, cemb])
        h, _, _ = self.self_attention([h, h])  # Image to Image
        h = self.resnet2([h, cemb])

        return h


class UpBlock(tf.keras.layers.Layer):
    """Add Residual block + Attention"""

    def __init__(
        self,
        out_channels,
        use_self_attention=False,
        use_cross_attention=False,
        attention_emb=None,
        num_res_blocks=1,
        use_upsample=False,
        use_scale_shift_norm=True,
        name='up_block',
        dtype=tf.float32,
        **kwargs,
    ):
        """
        Args:
        out_channels (int): Number of channels to project the images to

        """
        super(UpBlock, self).__init__(name=name, dtype=dtype, **kwargs)

        self.use_upsample = use_upsample
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self._use_scale_shift_norm = use_scale_shift_norm

        self.upsample_block = tf.identity
        self.resnet_blocks = []
        for resnet_counter in range(num_res_blocks):
            self.resnet_blocks.append(
                ResNetBlock(
                    out_channels, use_scale_shift_norm=use_scale_shift_norm, name='resnet_{}'.format(resnet_counter)
                )
            )

        if use_upsample:
            self.upsample_block = tf.keras.layers.UpSampling2D((2, 2), name='upsample')
            self.upsample_conv2d = tf.keras.layers.Conv2D(
                out_channels, kernel_size=(3, 3), strides=(1, 1), use_bias=True, padding='SAME', name='conv2d_upsample'
            )
        if use_self_attention:
            self.self_attn_block = ImageSelfAttention()
        if use_cross_attention:
            self.cross_attention = ImageTextCrossAttention()

    def call(self, inputs):
        """Call

        [image, cemb, text_token_embeddings, text_input_mask]
        """
        image, cemb, text_token_embeddings, text_input_mask = inputs
        h = image
        for resnet_block in self.resnet_blocks:
            h = resnet_block([h, cemb])
        h = self.upsample_block(h)
        if self.use_upsample:
            h = self.upsample_conv2d(h)

        if self.use_self_attention:
            h, _, _ = self.self_attn_block([h, h])  # Image to Image
        if self.use_cross_attention:
            h, _, _ = self.cross_attention([h, text_token_embeddings, text_input_mask])

        return h


class UnetModel(LegacyLayer):
    def __init__(
        self,
        text_embedding_dimension,
        time_embedding_dimension,
        input_channels=3,
        out_channels=128,
        channel_mult=[1, 2, 3, 4],
        use_self_attention=[True, True, True, True],
        use_cross_attention=[True, True, True, True],
        activation='swish',
        num_res_blocks=3,
        name: str = "unet",
        image_height=64,
        image_width=64,
        batch_size=None,
        sequence_length=None,
        use_dropout: bool = False,
        is_training: bool = False,
        use_scale_shift_norm=True,
        **kwargs,
    ):

        # Asserting number of layers which requires attention is same as
        # number of resolution scaling list
        assert len(channel_mult) == len(use_self_attention) == len(use_cross_attention)

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.channel_mult = channel_mult
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.num_res_blocks = num_res_blocks
        self._is_training = is_training
        self._use_dropout = use_dropout
        self.activation = activation
        self._image_height = image_height
        self._image_width = image_width
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._text_emb_dim = text_embedding_dimension
        self._time_emb_dim = time_embedding_dimension
        self._model_name = name
        self._use_scale_shift_norm = use_scale_shift_norm
        super(UnetModel, self).__init__(
            is_training=self._is_training, use_dropout=self._use_dropout, name=name, **kwargs
        )
        self._config_dict = {"is_training": self._is_training}

        # Define Layers
        # Time Embedding Layer
        self.time_embed_projection_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(out_channels, activation="swish", name='time_projection'),
                tf.keras.layers.Dense(out_channels),
            ]
        )

        self.time_embed_projection_layer = tf.keras.layers.Dense(
            out_channels, activation="swish", name='time_projection'
        )
        self.time_embed_projection_layer2 = tf.keras.layers.Dense(
            out_channels,
        )

        # Text Embedding Projection Layer

        self.text_embed_projection_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(out_channels, activation="swish", name='text_projection'),
                tf.keras.layers.Dense(out_channels),
            ]
        )
        # Image Projection Layers
        self.image_projection_layer = tf.keras.layers.Conv2D(
            out_channels, kernel_size=(3, 3), strides=(1, 1), use_bias=True, padding='SAME'
        )

        # Down blocks
        num_resolutions = len(self.channel_mult)

        # Down block Layer
        self.d_blocks = []
        # First block without downsample
        self.d_blocks.append(
            DownBlock(
                out_channels,
                use_self_attention=False,
                use_cross_attention=False,
                use_downsample=False,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )

        for index, ch_mult in enumerate(channel_mult):
            use_downsample = True
            if index == len(channel_mult) - 1:
                use_downsample = False
            current_out_channel = ch_mult * out_channels
            self.d_blocks.append(
                DownBlock(
                    current_out_channel,
                    use_self_attention=use_self_attention[index],
                    use_cross_attention=use_cross_attention[index],
                    use_downsample=use_downsample,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )

        # Middle block layer
        self.middle_block = MiddleBlock(current_out_channel)

        # # Up block Layer
        # We need to reverse it for up block and we have to skip the last one
        # Because otherwise dimension mismatch will happen, when we
        # contancate down block with up block hidden states
        channel_mult_up_layer = channel_mult[::-1][1:] + [1]
        self.u_blocks = []
        self.u_blocks.append(
            UpBlock(
                out_channels,
                use_self_attention=False,
                use_cross_attention=False,
                use_upsample=False,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        for index, ch_mult in enumerate(channel_mult_up_layer):
            use_upsample = True
            current_out_channel = ch_mult * out_channels
            if index == num_resolutions - 1:
                use_upsample = False
            self.u_blocks.append(
                UpBlock(
                    current_out_channel,
                    use_self_attention=False,
                    use_cross_attention=False,
                    use_upsample=use_upsample,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )

        # Last part
        self.last_group_norm = tfa.layers.GroupNormalization(name='last_group_norm')
        self.last_activation = tf.keras.activations.get(activation)
        self.last_conv2d = tf.keras.layers.Conv2D(
            input_channels, kernel_size=(3, 3), strides=(1, 1), use_bias=True, padding='SAME', name='last-conv2d'
        )

        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def get_model(self: LegacyLayer, initialize_only: bool = False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: Model layer
            initialize_only: If False, model (LegacyModel) wont be returned.
        """

        input_pixels = tf.keras.layers.Input(
            shape=(self._image_height, self._image_width, self.input_channels),
            batch_size=self._batch_size,
            dtype=tf.float32,
            name="input_pixels",
        )
        text_input_mask = tf.keras.layers.Input(
            shape=(self._sequence_length,),
            batch_size=self._batch_size,
            dtype=tf.int32,
            name="text_input_mask",
        )
        text_token_embeddings = tf.keras.layers.Input(
            shape=(self._sequence_length, self._text_emb_dim),
            batch_size=self._batch_size,
            dtype=tf.float32,
            name="text_token_embeddings",
        )
        sentence_embeddings = tf.keras.layers.Input(
            shape=(self._text_emb_dim,),
            batch_size=self._batch_size,
            dtype=tf.float32,
            name="sentence_embeddings",
        )
        time_embeddings = tf.keras.layers.Input(
            shape=(self._time_emb_dim,),
            batch_size=self._batch_size,
            dtype=tf.float32,
            name="time_embeddings",
        )
        inputs = {}
        inputs['input_pixels'] = input_pixels  # B x H x W x C
        inputs['time_embeddings'] = time_embeddings  # B x emb
        inputs['text_token_embeddings'] = text_token_embeddings  # B x S x emb_dim
        inputs['sentence_embeddings'] = sentence_embeddings  # B x emb_dim
        inputs['text_input_mask'] = text_input_mask  # B x S

        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        # Adding model_config is a hack
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        model.model_config = self._config_dict
        return model

    def call(self, inputs):
        """Forward Pass"""
        image = inputs['input_pixels']  # B x H x W x C
        time_embeddings = inputs['time_embeddings']  # B x emb
        text_token_embeddings = inputs['text_token_embeddings']  # B x S x emb_dim
        sentence_embeddings = inputs['sentence_embeddings']  # B x emb_dim
        text_input_mask = inputs['text_input_mask']  # B x S

        # time embeddings ( B x out_channels )
        time_embeddings_projected = self.time_embed_projection_layer(time_embeddings)
        # text embeddings ( B x out_channels )
        sentence_embeddings_projected = self.text_embed_projection_layer(sentence_embeddings)
        # image embeddings (B x H x W x out_channels)
        image_embeddings = self.image_projection_layer(image)

        # Conditional embeddings
        cemb = time_embeddings_projected + sentence_embeddings_projected
        h = image_embeddings

        # Downblock
        hs = []
        for block in self.d_blocks:

            h = block([h, cemb, text_token_embeddings, text_input_mask])
            hs.append(h)

        # Middleblock
        h = self.middle_block([h, cemb])

        # UpBlock
        for block in self.u_blocks:
            h_from_downblock = hs.pop()
            h_concatanated = tf.concat([h, h_from_downblock], axis=-1)
            h = block([h_concatanated, cemb, text_token_embeddings, text_input_mask])
        # End
        h = self.last_group_norm(h)
        h = self.last_activation(h)
        h = self.last_conv2d(h)

        return {'h': h}
