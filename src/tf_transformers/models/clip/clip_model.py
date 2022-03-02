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
"""The main wrapper around CLIP"""
from typing import Dict, Optional, Union

from absl import logging

from tf_transformers.core import ModelWrapper
from tf_transformers.core.read_from_hub import (
    get_config_cache,
    get_config_only,
    load_pretrained_model,
)
from tf_transformers.models.clip.clip import CLIPEncoder as Encoder
from tf_transformers.models.clip.clip_image_encoder import CLIPImageEncoder
from tf_transformers.models.clip.clip_text_encoder import CLIPTextEncoder
from tf_transformers.models.clip.configuration_clip import CLIPImageConfig
from tf_transformers.models.clip.configuration_clip import (
    CLIPImageConfig as ModelConfig,
)
from tf_transformers.models.clip.configuration_clip import CLIPTextConfig
from tf_transformers.models.clip.convert import convert_clip_pt as convert_pt
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import ENCODER_PRETRAINED_DOCSTRING

MODEL_TO_HF_URL = {
    "clip-vit-base-patch16": "tftransformers/clip-vit-base-patch16",
    "clip-vit-base-patch32": "tftransformers/clip-vit-base-patch32",
    "clip-vit-large-patch14": "tftransformers/clip-vit-large-patch14",
    "openai/clip-vit-base-patch16": "tftransformers/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32": "tftransformers/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14": "tftransformers/clip-vit-large-patch14",
}

code_example = r'''

        >>> from tf_transformers.models import  CLIPFeatureExtractorTF
        >>> from tf_transformers.models import  CLIPModel
        >>> image_path_list = # List fo image paths
        >>> model_name = 'openai/clip-base-patch16'
        >>> feature_extractor = CLIPFeatureExtractorTF(img_height=224, img_width=224)
        >>> model = CLIPModel.from_pretrained(model_name)

'''


class CLIPModel(ModelWrapper):
    """CLIP Encoder Wrapper"""

    def __init__(
        self, model_name: str = 'clip', cache_dir: Union[str, None] = None, save_checkpoint_cache: bool = True
    ):
        """
        Args:
            model_name (str): Model name
            cache_dir (str): cache dir to save the mode checkpoints
        """
        super(CLIPModel, self).__init__(
            model_name=model_name, cache_dir=cache_dir, save_checkpoint_cache=save_checkpoint_cache
        )

    def update_config(self, tft_config, hf_config):
        """Update tft config with hf config.

        Args:
            tft_config ([type]): [description]
            hf_config ([type]): [description]
        """
        # Add Vision config
        clip_config_hf_image = hf_config['vision_config']
        clip_text_config_hf = hf_config['text_config']

        clip_config_image = tft_config['vision_config']
        clip_text_config = tft_config['text_config']

        clip_config_image['image_size'] = clip_config_hf_image['image_size']
        clip_config_image['type_vocab_size'] = -1  # No type embeddings
        clip_config_image['num_attention_heads'] = clip_config_hf_image['num_attention_heads']
        clip_config_image['layer_norm_epsilon'] = clip_config_hf_image['layer_norm_eps']
        clip_config_image['intermediate_act'] = clip_config_hf_image['hidden_act']
        clip_config_image['hidden_act'] = clip_config_hf_image['hidden_act']
        clip_config_image['patch_size'] = clip_config_hf_image['patch_size']
        clip_config_image['num_hidden_layers'] = clip_config_hf_image['num_hidden_layers']
        clip_config_image['intermediate_size'] = clip_config_hf_image['intermediate_size']
        clip_config_image['embedding_size'] = clip_config_hf_image['hidden_size']
        clip_config_image['attention_head_size'] = (
            clip_config_image['embedding_size'] // clip_config_image['num_attention_heads']
        )
        clip_config_image['projection_dim'] = hf_config['projection_dim']
        # Add text config
        clip_text_config['vocab_size'] = clip_text_config_hf['vocab_size']
        clip_text_config['embedding_size'] = clip_text_config_hf['hidden_size']
        clip_text_config['intermediate_size'] = clip_text_config_hf['intermediate_size']
        clip_text_config['num_hidden_layers'] = clip_text_config_hf['num_hidden_layers']
        clip_text_config['max_position_embeddings'] = clip_text_config_hf['max_position_embeddings']
        clip_text_config['layer_norm_epsilon'] = clip_text_config_hf['layer_norm_eps']
        clip_text_config['num_attention_heads'] = clip_text_config_hf['num_attention_heads']
        clip_text_config['projection_dim'] = hf_config['projection_dim']

        tft_config['vision_config'] = clip_config_image
        tft_config['text_config'] = clip_text_config

        return tft_config

    @classmethod
    def get_config(cls, model_name: str):
        """Get a config from Huggingface hub if present"""

        # Check if it is under tf_transformers
        if model_name in MODEL_TO_HF_URL:
            URL = MODEL_TO_HF_URL[model_name]
            config_dict = get_config_only(URL)
            return config_dict
        else:
            config_image = CLIPImageConfig()
            config_text = CLIPTextConfig()

            vision_config = config_image.to_dict()
            text_config = config_text.to_dict()

            config_dict = {}
            config_dict['vision_config'] = vision_config
            config_dict['text_config'] = text_config

            cls_ref = cls()
            try:
                from transformers import PretrainedConfig

                hf_config = PretrainedConfig.from_pretrained(model_name)
                hf_config = hf_config.to_dict()
                config_dict = cls_ref.update_config(config_dict, hf_config)
                return config_dict
            except Exception as e:
                logging.info("Error: {}".format(e))
                logging.info("Failed loading config from HuggingFace")

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        vision_kwargs: Optional[Dict] = None,
        text_kwargs: Optional[Dict] = None,
        return_layer: bool = False,
        is_training=False,
        use_dropout=False,
        **kwargs,
    ):
        if isinstance(config, ModelConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config  # Dummy call to cls, as we need `_update_kwargs_and_config` function to be used here.
        cls_ref = cls()
        # if we allow names other than
        # whats in the class, we might not be able
        # to convert from hf properly.
        if "name" in kwargs:
            del kwargs["name"]

        vision_config = config_dict['vision_config']
        text_config = config_dict['text_config']

        vision_kwargs_copy = {}
        if vision_kwargs:
            if not isinstance(vision_kwargs, dict):
                raise ValueError("vision kwargs should be dict")
            vision_kwargs_copy = cls_ref._update_kwargs_and_config(vision_kwargs, vision_config)

        text_kwargs_copy = {}
        if text_kwargs:
            if not isinstance(text_kwargs, dict):
                raise ValueError("text kwargs should be dict")
            text_kwargs_copy = cls_ref._update_kwargs_and_config(text_kwargs, text_config)

        vision_encoder = CLIPImageEncoder(
            config=vision_config,
            name="clip_image",
            is_training=is_training,
            use_dropout=use_dropout,
            **vision_kwargs_copy,
        )
        text_encoder = CLIPTextEncoder(
            config=text_config, name="clip_text", is_training=is_training, use_dropout=use_dropout, **text_kwargs_copy
        )

        model_layer = Encoder(vision_encoder, text_encoder, is_training=is_training, use_dropout=use_dropout, **kwargs)
        model = model_layer.get_model()
        logging.info("Create model from config")
        if return_layer:
            return model_layer
        return model

    @classmethod
    @add_start_docstrings(
        "CLIP Model Pretrained with example :",
        ENCODER_PRETRAINED_DOCSTRING.format(
            "tf_transformers.models.CLIPModel", "tf_transformers.models.CLIPEncoder", "clip-base-patch16", code_example
        ),
    )
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: Union[str, None] = None,
        model_checkpoint_dir: Optional[str] = None,
        convert_from_hf: bool = True,
        return_layer: bool = False,
        return_config: bool = False,
        convert_fn_type: Optional[str] = "both",
        vision_kwargs: Optional[Dict] = None,
        text_kwargs: Optional[Dict] = None,
        save_checkpoint_cache: bool = True,
        load_from_cache: bool = True,
        skip_hub=False,
        is_training=False,
        use_dropout=False,
        **kwargs,
    ):
        # Load a base config and then overwrite it
        cls_ref = cls(model_name, cache_dir, save_checkpoint_cache)
        # Check if model is in out Huggingface cache
        if model_name in MODEL_TO_HF_URL and skip_hub is False:
            URL = MODEL_TO_HF_URL[model_name]
            config_dict, local_cache = get_config_cache(URL)
            vision_config = config_dict['vision_config']
            text_config = config_dict['text_config']

            vision_kwargs_copy = {}
            if vision_kwargs:
                if not isinstance(vision_kwargs, dict):
                    raise ValueError("vision kwargs should be dict")
                vision_kwargs_copy = cls_ref._update_kwargs_and_config(vision_kwargs, vision_config)

            text_kwargs_copy = {}
            if text_kwargs:
                if not isinstance(text_kwargs, dict):
                    raise ValueError("text kwargs should be dict")
                text_kwargs_copy = cls_ref._update_kwargs_and_config(text_kwargs, text_config)

            vision_encoder = CLIPImageEncoder(
                config=vision_config,
                name="clip_image",
                is_training=is_training,
                use_dropout=use_dropout,
                **vision_kwargs_copy,
            )
            text_encoder = CLIPTextEncoder(
                config=text_config,
                name="clip_text",
                is_training=is_training,
                use_dropout=use_dropout,
                **text_kwargs_copy,
            )

            model_layer = Encoder(
                vision_encoder, text_encoder, is_training=is_training, use_dropout=use_dropout, **kwargs
            )
            model = model_layer.get_model()
            # Load Model
            load_pretrained_model(model, local_cache, URL)
            if return_layer:
                if return_config:
                    return model_layer, config_dict
                return model_layer
            if return_config:
                return model, config_dict
            return model

        config_image = CLIPImageConfig()
        config_text = CLIPTextConfig()

        vision_config = config_image.to_dict()
        text_config = config_text.to_dict()

        config_dict = {}
        config_dict['vision_config'] = vision_config
        config_dict['text_config'] = text_config

        try:
            from transformers import PretrainedConfig

            hf_config = PretrainedConfig.from_pretrained(model_name)
            hf_config = hf_config.to_dict()
            config_dict = cls_ref.update_config(config_dict, hf_config)
        except Exception as e:
            logging.info("Error: {}".format(e))
            logging.info("Failed loading config from HuggingFace")

        # if we allow names other than
        # whats in the class, we might not be able
        # to convert from hf properly.
        if "name" in kwargs:
            del kwargs["name"]

        vision_kwargs_copy = {}
        if vision_kwargs:
            if not isinstance(vision_kwargs, dict):
                raise ValueError("vision kwargs should be dict")
            vision_kwargs_copy = cls_ref._update_kwargs_and_config(vision_kwargs, vision_config)

        text_kwargs_copy = {}
        if text_kwargs:
            if not isinstance(text_kwargs, dict):
                raise ValueError("text kwargs should be dict")
            text_kwargs_copy = cls_ref._update_kwargs_and_config(text_kwargs, text_config)

        vision_encoder = CLIPImageEncoder(
            config=vision_config,
            name="clip_image",
            is_training=is_training,
            use_dropout=use_dropout,
            **vision_kwargs_copy,
        )
        text_encoder = CLIPTextEncoder(
            config=text_config, name="clip_text", is_training=is_training, use_dropout=use_dropout, **text_kwargs_copy
        )

        model_layer = Encoder(vision_encoder, text_encoder, is_training=is_training, use_dropout=use_dropout, **kwargs)
        model = model_layer.get_model()

        # Give preference to model_checkpoint_dir
        if model_checkpoint_dir:
            model.load_checkpoint(model_checkpoint_dir)
        else:
            load_succesfuly = False
            if cls_ref.model_path.exists():
                try:
                    if load_from_cache:
                        model.load_checkpoint(str(cls_ref.model_path))
                        load_succesfuly = True
                except Exception as e:
                    logging.warn(e)
            if convert_from_hf and not load_succesfuly:
                if convert_fn_type == "both":
                    cls_ref.convert_hf_to_tf(
                        model,
                        config_dict,
                        convert_tf_fn=None,
                        convert_pt_fn=convert_pt,
                    )
                if convert_fn_type == "pt":
                    cls_ref.convert_hf_to_tf(model, config_dict, convert_tf_fn=None, convert_pt_fn=convert_pt)

        if return_layer:
            if return_config:
                return model_layer, config_dict
            return model_layer
        if return_config:
            return model, config_dict
        return model
