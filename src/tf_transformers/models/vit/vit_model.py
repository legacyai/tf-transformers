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
"""The main wrapper around ViT"""
from typing import Optional, Union

from absl import logging

from tf_transformers.core import ModelWrapper
from tf_transformers.core.read_from_hub import (
    get_config_cache,
    get_config_only,
    load_pretrained_model,
)
from tf_transformers.models.vit import ViTEncoder as Encoder
from tf_transformers.models.vit.configuration_vit import ViTConfig as ModelConfig
from tf_transformers.models.vit.convert import convert_vit_pt as convert_pt
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    ENCODER_MODEL_CONFIG_DOCSTRING,
    ENCODER_PRETRAINED_DOCSTRING,
)

MODEL_TO_HF_URL = {}

code_example = r'''

        >>> from tf_transformers.models import  ViTFeatureExtractorTF
        >>> from tf_transformers.models import  VitModel
        >>> image_path_list = # List fo image paths
        >>> model_name = 'google/vit-base-patch16-224'
        >>> vit_feature_extractor_tf = ViTFeatureExtractorTF(img_height=224, img_width=224)
        >>> feature_extractor = ViTFeatureExtractorTF(img_height=224, img_width=224)
        >>> model = ViTModel.from_pretrained(model_name, classification_labels=1000)
        >>> input_features = feature_extractor({'image': tf.constant(image_path_list)})
        >>> model_outputs = model(input_features)
        >>> predicted_class = tf.argmax(model_outputs['class_logits'], axis=-1)


'''


class ViTModel(ModelWrapper):
    """vit Encoder Wrapper"""

    def __init__(self, model_name: str = 'vit', cache_dir: Union[str, None] = None, save_checkpoint_cache: bool = True):
        """
        Args:
            model_name (str): Model name
            cache_dir (str): cache dir to save the mode checkpoints
        """
        super(ViTModel, self).__init__(
            model_name=model_name, cache_dir=cache_dir, save_checkpoint_cache=save_checkpoint_cache
        )

    def update_config(self, tft_config, hf_config):
        """Update tft config with hf config.

        Args:
            tft_config ([type]): [description]
            hf_config ([type]): [description]
        """
        tft_config["image_size"] = hf_config["image_size"]
        tft_config["patch_size"] = hf_config["patch_size"]
        tft_config["num_channels"] = hf_config["num_channels"]
        tft_config["embedding_size"] = hf_config["hidden_size"]
        tft_config["intermediate_size"] = hf_config["intermediate_size"]

        tft_config["num_attention_heads"] = hf_config["num_attention_heads"]
        tft_config["num_hidden_layers"] = hf_config["num_hidden_layers"]

        try:
            tft_config["num_labels"] = len(hf_config["id2label"])
        except:
            pass

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
            # Check inside huggingface
            config = ModelConfig()
            config_dict = config.to_dict()
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
    @add_start_docstrings(
        "ViT Model from config :",
        ENCODER_MODEL_CONFIG_DOCSTRING.format("transformers.models.VitEncoder", "tf_transformers.models.vit.ViTConfig"),
    )
    def from_config(cls, config: ModelConfig, return_layer: bool = False, **kwargs):
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

        kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)

        # if a config is provided, we wont be doing any extra .
        # Just create a model and return it with random_weights
        # (Distribute strategy fails)
        model_layer = Encoder(config_dict, **kwargs_copy)
        model = model_layer.get_model()
        logging.info("Create model from config")
        if return_layer:
            return model_layer
        return model

    @classmethod
    @add_start_docstrings(
        "Bert Model Pretrained with example :",
        ENCODER_PRETRAINED_DOCSTRING.format(
            "tf_transformers.models.BertModel", "tf_transformers.models.BertEncoder", "bert-base-uncased", code_example
        ),
    )
    def from_pretrained(
        cls,
        model_name: str,
        classification_labels: int = None,
        cache_dir: Union[str, None] = None,
        model_checkpoint_dir: Optional[str] = None,
        convert_from_hf: bool = True,
        return_layer: bool = False,
        return_config: bool = False,
        convert_fn_type: Optional[str] = "both",
        save_checkpoint_cache: bool = True,
        load_from_cache: bool = True,
        skip_hub=False,
        **kwargs,
    ):
        # Load a base config and then overwrite it
        cls_ref = cls(model_name, cache_dir, save_checkpoint_cache)
        # Check if model is in out Huggingface cache
        if model_name in MODEL_TO_HF_URL and skip_hub is False:
            URL = MODEL_TO_HF_URL[model_name]
            config_dict, local_cache = get_config_cache(URL)
            kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)
            if classification_labels:
                logging.info("Using pretrained classifier layer with num_class={}".format(classification_labels))
            model_layer = Encoder(config_dict, classification_labels=classification_labels, **kwargs_copy)

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

        config = ModelConfig()
        config_dict = config.to_dict()

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

        kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)
        if classification_labels:
            logging.info("Using pretrained classifier layer with num_class={}".format(classification_labels))
        model_layer = Encoder(config_dict, classification_labels=classification_labels, **kwargs_copy)
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
                return model_layer, config
            return model_layer
        if return_config:
            return model, config
        return model
