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
"""Base Diffusion Model condtioned on time and text"""
from typing import Dict

import tensorflow as tf
from beta_schedule import get_beta_schedule
from time_embedding_layers import TimeEmbedding

from tf_transformers.core import LegacyLayer, LegacyModel


class BaseDiffusion(LegacyLayer):
    def __init__(
        self,
        config: Dict,
        text_encoder_model,
        unet_model,
        name='diffusion',
        batch_size=None,
        sequence_length=None,
        use_dropout: bool = False,
        is_training: bool = False,
        **kwargs,
    ):
        beta_schedule = config['beta_schedule']
        num_diffusion_steps = config['diffusion_steps']
        time_emb_dimension = config['time_emb_dimension']
        # image_height = config['image_height']
        # image_width = config['image_width']
        # input_channels = config['input_channels']

        self._batch_size = batch_size
        self._sequence_length = sequence_length

        self._is_training = is_training
        self._use_dropout = use_dropout
        self._model_name = name

        super(BaseDiffusion, self).__init__(
            is_training=self._is_training, use_dropout=self._use_dropout, name=name, **kwargs
        )
        self._config_dict = {"is_training": self._is_training}
        self._config_dict.update(config)

        betas = get_beta_schedule(beta_schedule=beta_schedule, num_diffusion_timesteps=num_diffusion_steps)
        betas = tf.constant(betas)
        alphas = 1.0 - betas
        self.alphas_cumprod = tf.math.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = tf.concat([[1.0], self.alphas_cumprod[:-1]], axis=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.math.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.math.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = tf.math.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.math.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.math.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = tf.math.log(
            tf.concat([[self.posterior_variance[1]], self.posterior_variance[1:]], axis=0)
        )
        self.posterior_mean_coef1 = betas * tf.math.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * tf.math.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        # Time Embedding Layer
        self.time_emb_dimension = time_emb_dimension
        self.time_embedding_layer = TimeEmbedding(n_channels=time_emb_dimension)

        # Text Encoder
        self.text_encoder = text_encoder_model
        self.unet_model = unet_model

        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def get_model(self: LegacyLayer, initialize_only: bool = False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: Model layer
            initialize_only: If False, model (LegacyModel) wont be returned.
        """

        input_pixels = tf.keras.layers.Input(
            shape=(
                self._config_dict['image_height'],
                self._config_dict['image_width'],
                self._config_dict['input_channels'],
            ),
            batch_size=self._batch_size,
            dtype=tf.float32,
            name="input_pixels",
        )
        input_noise = tf.keras.layers.Input(
            shape=(
                self._config_dict['image_height'],
                self._config_dict['image_width'],
                self._config_dict['input_channels'],
            ),
            batch_size=self._batch_size,
            dtype=tf.float32,
            name="input_noise",
        )
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

        time_steps = tf.keras.layers.Input(
            shape=(self._batch_size,),
            batch_size=1,
            dtype=tf.int32,
            name="time_steps",
        )

        inputs = {}
        inputs['input_pixels'] = input_pixels  # B x H x W x C
        inputs['input_ids'] = input_ids  # B x emb
        inputs['input_mask'] = input_mask  # B x S x emb_dim
        inputs['time_steps'] = time_steps  # B x emb_dim
        inputs['noise'] = input_noise

        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        # Adding model_config is a hack
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        model.model_config = self._config_dict
        return model

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs = tf.shape(t)[0]
        # assert x_shape[0] == bs
        out = tf.cast(tf.gather(a, t), tf.float32)
        # assert out.shape == [bs]
        return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        # assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_mean_embeddings(self, token_embeddings, input_mask):
        """
        Mean embeddings
        """
        # mask PAD tokens
        token_emb_masked = token_embeddings * tf.cast(tf.expand_dims(input_mask, 2), tf.float32)
        total_non_padded_tokens_per_batch = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.float32)
        # Convert to 2D
        total_non_padded_tokens_per_batch = tf.expand_dims(total_non_padded_tokens_per_batch, 1)
        mean_embeddings = tf.reduce_sum(token_emb_masked, axis=1) / total_non_padded_tokens_per_batch
        return mean_embeddings

    def call(self, inputs):
        """ """

        x_start = inputs['input_pixels']  # B x H x W x C
        input_mask = inputs['input_mask']  # B x S
        input_ids = inputs['input_ids']  # B x S
        time_steps = tf.squeeze(inputs['time_steps'], axis=0)  # 1 x d -> d
        noise = inputs['noise']

        # Get text token embeddings
        text_inputs = {'input_ids': input_ids, 'input_mask': input_mask}
        text_result = self.text_encoder(text_inputs)
        text_token_embeddings = text_result['token_embeddings']
        if 'sentence_embeddings' not in text_result:
            sentence_embeddings = self.get_mean_embeddings(text_token_embeddings, input_mask)
        else:
            sentence_embeddings = text_result['sentence_embeddings']

        # Get noise version of image at defined time step
        x_t = self.q_sample(x_start, time_steps, noise)

        # Get time embeddings
        time_embeddings = self.time_embedding_layer(time_steps)

        # Unet inputs
        unet_inputs = {}
        unet_inputs['input_pixels'] = x_t  # B x H x W x C
        unet_inputs['time_embeddings'] = time_embeddings  # B x emb
        unet_inputs['text_token_embeddings'] = text_token_embeddings  # B x S x emb_dim
        unet_inputs['sentence_embeddings'] = sentence_embeddings  # B x emb_dim
        unet_inputs['text_input_mask'] = input_mask  # B x S

        unet_result = self.unet_model(unet_inputs)

        result = {}
        result.update(unet_result)
        result['noise'] = noise
        return result
