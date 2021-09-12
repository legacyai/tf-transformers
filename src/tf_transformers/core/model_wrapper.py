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
from typing import Callable, Dict, Union

import tensorflow as tf
from absl import logging

logging.set_verbosity("INFO")

_PREFIX_DIR = "tf_transformers_cache"
HF_VERSION = "4.6.0"


class ModelWrapper(ABC):
    """Model Wrapper for all models"""

    def __init__(self, model_name: str, cache_dir: Union[str, None], save_checkpoint_cache: bool):
        """

        Args:
            model_name: Model name as per in HF Transformers
            cache_dir_path : Path to cache directory . Default is :obj:`/tmp/tf_transformers/cache`
            save_checkpoint_cache: :obj:`bool`. Whether to save the converted model in cache directory.
        """
        self.model_name = model_name
        self.save_checkpoint_cache = save_checkpoint_cache
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()

        self.cache_dir = Path(cache_dir, _PREFIX_DIR)
        self.create_cache_dir(self.cache_dir)
        self.model_path = Path(self.cache_dir, self.model_name)
        self.hf_version = HF_VERSION

    @abstractmethod
    def update_config(self, tft_config: Dict, hf_config: Dict):
        """This is custom to the models."""
        pass

    def _update_kwargs_and_config(self, kwargs: Dict, config: Dict):
        """keywords in kwargs has to be updated in  config
        Otherwise it wont be recongnized by the Model

        Args:
            kwargs : Keyword arguments for the model
            config : Model config


        """
        kwargs_copy = kwargs.copy()

        for key, value in kwargs.items():
            if key in config:
                config[key] = value
                del kwargs_copy[key]
        return kwargs_copy

    def create_cache_dir(self, cache_path: Path):
        """Create Cache Directory

        Args:
            cache_path : Path
        """
        if not cache_path.exists():  # If cache path not exists
            cache_path.mkdir()

    def convert_hf_to_tf(
        self,
        model: tf.keras.Model,
        config: Dict,
        convert_tf_fn: Union[Callable, None],
        convert_pt_fn: Union[Callable, None],
    ):
        """Convert HF to TFT

        Args:
            config: Dict
            convert_tf_fn: TF based conversion function from HF to TFT
            convert_pt_fn: PT based conversion function from HF to TFT
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

        if self.save_checkpoint_cache:
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
