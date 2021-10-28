import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from tf_transformers.data.utils import hf_dump_chars_to_textfile

# A logger for this file
log = logging.getLogger(__name__)


def write_data(cfg):
    """Load dataset and write to txt file"""
    output_file = cfg.data.output_text_file
    if os.path.isfile(output_file):
        raise FileExistsError()

    from datasets import load_dataset

    if cfg.data.version:
        dataset = load_dataset(cfg.data.name, cfg.data.version)
    else:
        dataset = load_dataset(cfg.data.name)

    split = cfg.data.split  # train, test, dev
    data_keys = cfg.data.keys  # text
    hf_dump_chars_to_textfile(output_file, dataset[split], data_keys, max_char=-1)


@hydra.main(config_path="config", config_name="data_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    write_data(cfg)


if __name__ == "__main__":
    run()
