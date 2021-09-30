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
"""This is the main script to run GLUE benchmark"""
import hydra
import pandas as pd
from cola import run_cola
from mnli import run_mnli
from mrpc import run_mrpc
from omegaconf import DictConfig
from qnli import run_qnli
from qqp import run_qqp
from rte import run_rte
from sst2 import run_sst2
from stsb import run_stsb


def flat_callbacks_to_df(history):
    callback_flatten = {}
    for item in history['callbacks']:
        for k, v in item[0].items():
            if k in callback_flatten:
                callback_flatten[k].append(v)
            else:
                callback_flatten[k] = [v]
    df = pd.DataFrame(callback_flatten)
    return df


def run_glue(cfg):
    """All GLUE experiments starts here.
    We will train each task one by one.
    """

    # Run MRPC
    history = run_mrpc(cfg)
    df_mrpcs = flat_callbacks_to_df(history)  # noqa
    df_mrpcs.to_csv("mrpc_eval.csv", index=False)

    # Run MNLI
    history = run_mnli(cfg)
    df_mnli = flat_callbacks_to_df(history)  # noqa
    df_mnli.to_csv("mnli_eval.csv", index=False)

    # Run COLA
    history = run_cola(cfg)
    df_cola = flat_callbacks_to_df(history)  # noqa
    df_cola.to_csv("cola_eval.csv", index=False)

    # Run QNLI
    history = run_qnli(cfg)
    df_qnli = flat_callbacks_to_df(history)  # noqa
    df_qnli.to_csv("qnli_eval.csv", index=False)

    # Run QQP
    history = run_qqp(cfg)
    df_qqp = flat_callbacks_to_df(history)  # noqa
    df_qqp.to_csv("qqp_eval.csv", index=False)

    # Run RTE
    history = run_rte(cfg)
    df_rte = flat_callbacks_to_df(history)  # noqa
    df_rte.to_csv("rte_eval.csv", index=False)

    # Run SST2
    history = run_sst2(cfg)
    df_sst2 = flat_callbacks_to_df(history)  # noqa
    df_sst2.to_csv("sst2_eval.csv", index=False)

    # Run STSB
    history = run_stsb(cfg)
    df_stsb = flat_callbacks_to_df(history)  # noqa
    df_stsb.to_csv("stsb_eval.csv", index=False)

    return True


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # For TPU static padding is useful.
    if cfg.trainer.strategy == 'tpu':
        if cfg.data.static_padding is False:
            raise ValueError(
                "Make sure `static_padding` is set to True, when strategy is tpu. Please check config.yaml"
            )

    cfg_dict = dict(cfg)
    # If a specific task is specified, it must be under "glue" key
    # If not we will run the whole process
    if "glue" not in cfg_dict:
        run_glue(cfg)
    else:
        # Run mrpc
        if "mrpc" in cfg_dict["glue"]["task"]["name"]:
            run_mrpc(cfg)

        # Run mnli
        if "mnli" in cfg_dict["glue"]["task"]["name"]:
            run_mnli(cfg)

        # Run cola
        if "cola" in cfg_dict["glue"]["task"]["name"]:
            run_cola(cfg)

        # Run qnli
        if "qnli" in cfg_dict["glue"]["task"]["name"]:
            run_qnli(cfg)

        # Run qqp
        if "qqp" in cfg_dict["glue"]["task"]["name"]:
            run_qqp(cfg)

        # Run rte
        if "rte" in cfg_dict["glue"]["task"]["name"]:
            run_rte(cfg)

        # Run sst2
        if "sst2" in cfg_dict["glue"]["task"]["name"]:
            run_sst2(cfg)

        # Run stsb
        if "stsb" in cfg_dict["glue"]["task"]["name"]:
            run_stsb(cfg)


if __name__ == "__main__":
    run()
