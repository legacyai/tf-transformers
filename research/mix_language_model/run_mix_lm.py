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
"""This is the main script to run Mix Language Model"""
import os
import warnings

import hydra
from absl import logging
from omegaconf import DictConfig
from train_mix_lm import run_train

logging.set_verbosity("INFO")

# We set PROJECT_NAME from ENVIORNMENT VARIABLE
WANDB_PROJECT = os.getenv('WANDB_PROJECT', None)
use_wandb = True
if WANDB_PROJECT is None:
    warnings.warn(
        "For wandb-project should not be None.\
        Set export WANDB_PROJECT=<project_name>"
    )
    use_wandb = False


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    print("Config", cfg)
    config_dict = dict(cfg)
    # For TPU, we need to initialize it before tf text dataset
    # starts triggering. Hack
    if cfg.trainer.strategy == 'tpu':
        from model import get_trainer

        distribution_strategy = 'tpu'
        num_gpus = 0
        tpu_address = cfg.trainer.tpu_address
        get_trainer(
            distribution_strategy=distribution_strategy,
            num_gpus=num_gpus,
            tpu_address=tpu_address,
            dtype=cfg.trainer.dtype,
        )  # noqa

    if use_wandb:
        import wandb

        wandb.init(project=WANDB_PROJECT, config=config_dict, sync_tensorboard=True)
        history = run_train(cfg, wandb)
    else:
        # Set wandb = None
        history = run_train(cfg, None)
    return history


if __name__ == "__main__":
    history = run()
