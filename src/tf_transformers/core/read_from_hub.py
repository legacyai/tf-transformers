import tensorflow as tf
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from absl import logging
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

def load_pretrained_model(model, local_cache, url):
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