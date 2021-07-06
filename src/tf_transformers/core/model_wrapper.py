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
"""ModelWrapper setup"""
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from absl import logging

logging.set_verbosity("INFO")

_PREFIX_DIR = "tf_transformers_cache"
HF_VERSION = "4.6.0"


class ModelWrapper(ABC):
    """Model Wrapper for all models"""

    def __init__(self, model_name, cache_dir):
        """

        Args:
            cache_dir ([str]): [None/ cache_dir string]
            model_name ([str]): [name of the model]
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        if cache_dir is None:
            self.cache_dir = tempfile.gettempdir()

        self.cache_dir = Path(self.cache_dir, _PREFIX_DIR)
        self.create_cache_dir(self.cache_dir)
        self.model_path = Path(self.cache_dir, self.model_name)
        self.hf_version = HF_VERSION

    @abstractmethod
    def update_config(self):
        pass

    def _update_kwargs_and_config(self, kwargs, config):
        """keywords in kwargs has to be updated oi  config
        Otherwise it wont be recongnized by the Model

        Args:
            kwargs (dict): Keyword arguments for the model
            config (dict): Model config

        Returns:
            [dict]: Updated kwargs
        """
        kwargs_copy = kwargs.copy()

        for key, value in kwargs.items():
            if key in config:
                config[key] = value
                del kwargs_copy[key]
        return kwargs_copy

    def create_cache_dir(self, cache_path):
        """Create Cache Directory

        Args:
            cache_path ([type]): [Path object]
        """
        if not cache_path.exists():  # If cache path not exists
            cache_path.mkdir()

    def convert_hf_to_tf(self, model, config, convert_tf_fn, convert_pt_fn, save_checkpoint_cache=True):
        """Convert TTF from HF

        Args:
            model ([tf.keras.Model]): [tf-transformer model]
            convert_fn ([function]): [Function which converts HF to TTF]
        """
        # HF has '-' , instead of '_'
        import transformers

        if self.hf_version:
            if transformers.__version__ != self.hf_version:
                logging.warning(
                    "Expected `transformers` version `{}`, but found version `{}`.\
        The conversion might or might not work.".format(
                        self.hf_version, transformers.__version__
                    )
                )
        hf_model_name = self.model_name
        convert_success = False
        if convert_tf_fn:
            try:
                convert_tf_fn(model, config, hf_model_name)
                convert_success = True
                logging.info("Successful ✅: Converted model using TF HF")
            except Exception as e:
                logging.error(e)
                logging.info("Failed ❌: Converted model using TF HF")

        if convert_success is False and convert_pt_fn:
            try:
                convert_pt_fn(model, config, hf_model_name)
                logging.info("Successful ✅: Converted model using PT HF")
                convert_success = True
            except Exception as e:
                logging.error(e)
                logging.info("Failed ❌: Converted model using PT HF")

        if save_checkpoint_cache:
            if convert_success:
                model.save_checkpoint(str(self.model_path), overwrite=True)
                logging.info(
                    "Successful ✅: Asserted and Converted `{}` from HF and saved it in cache folder {}".format(
                        hf_model_name, str(self.model_path)
                    )
                )
            else:
                model.save_checkpoint(str(self.model_path), overwrite=True)
                logging.info(
                    "Saved model in cache folder with randomly ❌ initialized values  {}".format(str(self.model_path))
                )
