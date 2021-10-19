import json
from pathlib import Path

import tensorflow as tf
from absl import logging
from huggingface_hub import hf_hub_download, snapshot_download

logging.set_verbosity("INFO")


def get_config_cache(url: str):
    """Load model from Huggingface hub"""
    local_cache = snapshot_download(repo_id=url)

    # Load config from cache
    config_path = Path(local_cache, "config.json")
    if not config_path.exists():
        raise ValueError("config.json is not present in model hub {}".format(url))
    config_dict = json.load(open(config_path))
    return config_dict, local_cache


def get_config_only(url: str):
    """Load config from Huggingface hub"""
    config_path = hf_hub_download(repo_id=url, filename="config.json")
    # Load config from cache
    config_dict = json.load(open(config_path))
    return config_dict


def load_pretrained_model(model: tf.keras.Model, local_cache: str, url: str):
    """Load model from cache"""
    try:
        local_device_option = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    except:
        import traceback

        print(traceback.format_exc())
        local_device_option = tf.CheckpointOptions(experimental_io_device="/job:localhost")
    else:
        local_device_option = None

    model.load_checkpoint(local_cache, options=local_device_option)
    logging.info("Successful ✅: Loaded model from {}".format(url))
