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

import numpy as np
import tensorflow as tf
from beta_schedule import get_beta_schedule

from tf_transformers.core import LegacyLayer, LegacyModel


def noise_like(shape, noise_fn=tf.random.normal, repeat=False, dtype=tf.float32):
    repeat_noise = lambda: tf.repeat(noise_fn(shape=(1, *shape[1:]), dtype=dtype), repeats=shape[0], axis=0)
    noise = lambda: noise_fn(shape=shape, dtype=dtype)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(LegacyLayer):
    """
    Contains utilities for the diffusion model.
    """

    def __init__(
        self,
        config: Dict,
        unet_model,
        text_encoder_model=None,
        name='diffusion',
        batch_size=None,
        sequence_length=None,
        use_dropout: bool = False,
        is_training: bool = False,
        dtype=tf.float32,
        **kwargs,
    ):

        beta_schedule = config['beta_schedule']
        num_diffusion_steps = config['diffusion_steps']
        # image_height = config['image_height']
        # image_width = config['image_width']
        # input_channels = config['input_channels']
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        tf_dtype = dtype
        self._model_name = name

        self.text_encoder_model = text_encoder_model
        self.unet_model = unet_model

        self._is_training = is_training
        self._use_dropout = use_dropout
        super(GaussianDiffusion, self).__init__(
            is_training=self._is_training, use_dropout=self._use_dropout, name=name, **kwargs
        )
        self._config_dict = {"is_training": self._is_training}
        self._config_dict.update(config)

        betas = get_beta_schedule(beta_schedule=beta_schedule, num_diffusion_timesteps=num_diffusion_steps)
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == (timesteps,)

        self.betas = tf.constant(betas, dtype=tf_dtype)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf_dtype)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf_dtype)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf_dtype)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf_dtype)
        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1.0 - alphas_cumprod), dtype=tf_dtype)
        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod), dtype=tf_dtype)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf_dtype)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf_dtype)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf_dtype)
        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=tf_dtype
        )
        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=tf_dtype
        )

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

        time_steps = tf.keras.layers.Input(
            shape=(self._batch_size,),
            batch_size=1,
            dtype=tf.int32,
            name="time_steps",
        )

        inputs = {}
        inputs['input_pixels'] = input_pixels  # B x H x W x C
        inputs['time_steps'] = time_steps  # B x emb_dim
        inputs['noise'] = input_noise

        if self.text_encoder_model:
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
            inputs['input_ids'] = input_ids  # B x emb
            inputs['input_mask'] = input_mask  # B x S x emb_dim

        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        # Adding model_config is a hack
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        model.model_config = self._config_dict

        return model

    @staticmethod
    def get_mean_embeddings(token_embeddings, input_mask):
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

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs = tf.shape(t)[0]
        # assert x_shape[0] == bs
        out = tf.gather(a, t)
        # assert out.shape == [bs]
        return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = tf.random.normal(shape=x_start.shape)

        # assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        assert x_t.shape == noise.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # def p_losses(self, denoise_fn, x_start, t, noise=None):
    #     """
    #     Training loss calculation
    #     """
    #     B, H, W, C = x_start.shape.as_list()
    #     assert t.shape == [B]

    #     if noise is None:
    #         noise = tf.random_normal(shape=x_start.shape, dtype=x_start.dtype)
    #     assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     x_recon = denoise_fn(x_noisy, t)
    #     assert x_noisy.shape == x_start.shape
    #     assert x_recon.shape[:3] == [B, H, W] and len(x_recon.shape) == 4

    #     if self.loss_type == 'noisepred':
    #         # predict the noise instead of x_start. seems to be weighted naturally like SNR
    #         assert x_recon.shape == x_start.shape
    #         losses = nn.meanflat(tf.squared_difference(noise, x_recon))
    #     else:
    #         raise NotImplementedError(self.loss_type)

    #     assert losses.shape == [B]
    #     return losses

    def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool):
        if self.loss_type == 'noisepred':
            x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
        else:
            raise NotImplementedError(self.loss_type)

        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        assert model_mean.shape == x_recon.shape == x.shape
        assert posterior_variance.shape == posterior_log_variance.shape == [x.shape[0], 1, 1, 1]
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, denoise_fn, *, x, t, noise_fn, clip_denoised=True, repeat_noise=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, noise_fn, repeat_noise)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x.shape[0]] + [1] * (len(x.shape) - 1))
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, denoise_fn, *, shape, noise_fn=tf.random.normal):
        """
        Generate samples
        """
        i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
        assert isinstance(shape, (tuple, list))
        img_0 = noise_fn(shape=shape, dtype=tf.float32)
        _, img_final = tf.while_loop(
            cond=lambda i_, _: tf.greater_equal(i_, 0),
            body=lambda i_, img_: [
                i_ - 1,
                self.p_sample(denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn),
            ],
            loop_vars=[i_0, img_0],
            shape_invariants=[i_0.shape, img_0.shape],
            back_prop=False,
        )
        assert img_final.shape == shape
        return img_final

    def p_sample_loop_trajectory(self, denoise_fn, *, shape, noise_fn=tf.random.normal, repeat_noise_steps=-1):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
        assert isinstance(shape, (tuple, list))
        img_0 = noise_like(shape, noise_fn, repeat_noise_steps >= 0)
        times = tf.Variable([i_0])
        imgs = tf.Variable([img_0])
        # Steps with repeated noise
        times, imgs = tf.while_loop(
            cond=lambda times_, _: tf.less_equal(self.num_timesteps - times_[-1], repeat_noise_steps),
            body=lambda times_, imgs_: [
                tf.concat([times_, [times_[-1] - 1]], 0),
                tf.concat(
                    [
                        imgs_,
                        [
                            self.p_sample(
                                denoise_fn=denoise_fn,
                                x=imgs_[-1],
                                t=tf.fill([shape[0]], times_[-1]),
                                noise_fn=noise_fn,
                                repeat_noise=True,
                            )
                        ],
                    ],
                    0,
                ),
            ],
            loop_vars=[times, imgs],
            shape_invariants=[tf.TensorShape([None, *i_0.shape]), tf.TensorShape([None, *img_0.shape])],
            back_prop=False,
        )
        # Steps with different noise for each batch element
        times, imgs = tf.while_loop(
            cond=lambda times_, _: tf.greater_equal(times_[-1], 0),
            body=lambda times_, imgs_: [
                tf.concat([times_, [times_[-1] - 1]], 0),
                tf.concat(
                    [
                        imgs_,
                        [
                            self.p_sample(
                                denoise_fn=denoise_fn,
                                x=imgs_[-1],
                                t=tf.fill([shape[0]], times_[-1]),
                                noise_fn=noise_fn,
                                repeat_noise=False,
                            )
                        ],
                    ],
                    0,
                ),
            ],
            loop_vars=[times, imgs],
            shape_invariants=[tf.TensorShape([None, *i_0.shape]), tf.TensorShape([None, *img_0.shape])],
            back_prop=False,
        )
        assert imgs[-1].shape == shape
        return times, imgs

    def call(self, inputs):
        """ """

        x_start = inputs['input_pixels']  # B x H x W x C
        time_steps = inputs['time_steps']  # B x 1 -> B
        noise = inputs['noise']

        # Unet inputs
        unet_inputs = {}

        if self.text_encoder_model:
            input_mask = inputs['input_mask']  # B x S
            input_ids = inputs['input_ids']  # B x S
            # Get text token embeddings
            text_inputs = {'input_ids': input_ids, 'input_mask': input_mask}
            text_result = self.text_encoder_model(text_inputs)
            text_token_embeddings = text_result['token_embeddings']
            if 'sentence_embeddings' not in text_result:
                sentence_embeddings = self.get_mean_embeddings(text_token_embeddings, input_mask)
            else:
                sentence_embeddings = text_result['sentence_embeddings']

            unet_inputs['text_token_embeddings'] = text_token_embeddings  # B x S x emb_dim
            unet_inputs['sentence_embeddings'] = sentence_embeddings  # B x emb_dim
            unet_inputs['text_input_mask'] = input_mask  # B x S

        # Get noise version of image at defined time step
        x_t = self.q_sample(x_start, tf.squeeze(time_steps, axis=0), noise)

        unet_inputs['input_pixels'] = x_t  # B x H x W x C
        unet_inputs['time_steps'] = time_steps  # B x 1
        unet_result = self.unet_model(unet_inputs)

        result = {}
        result.update(unet_result)
        result['noise'] = noise

        return result
