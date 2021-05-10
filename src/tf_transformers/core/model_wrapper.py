import tempfile
from pathlib import Path
from absl import logging

logging.set_verbosity("INFO")

_PREFIX_DIR = "tf_transformers_cache"


class ModelWrapper:
    """Model Wrapper for all models"""

    def __init__(self, cache_dir, model_name):
        """

        Args:
            cache_dir ([str]): [None/ cache_dir string]
            model_name ([str]): [name of the model]
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        if cache_dir is None:
            self.cache_dir = tempfile.gettempdir()

        self.cache_dir = Path(self.cache_dir, _PREFIX_DIR)
        self.create_cache_dir(self.cache_dir)
        self.model_path = Path(self.cache_dir, self.model_name)

    def create_cache_dir(self, cache_path):
        """Create Cache Directory

        Args:
            cache_path ([type]): [Path object]
        """
        if not cache_path.exists():  # If cache path not exists
            cache_path.mkdir()

    def convert_hf_to_tf(self, model, convert_tf_fn, convert_pt_fn, hf_version="4.3.3"):
        """Convert TTF from HF

        Args:
            model ([tf.keras.Model]): [tf-transformer model]
            convert_fn ([function]): [Function which converts HF to TTF]
        """
        # HF has '-' , instead of '_'
        import transformers

        if transformers.__version__ != hf_version:
            raise ValueError(
                "Expected `transformers` version `{}`, but found version `{}`.".format(
                    hf_version, transformers.__version__
                )
            )
        hf_model_name = self.model_name
        convert_success = False
        if convert_tf_fn:
            try:
                convert_tf_fn(hf_model_name)
                convert_success = True
                logging.info("Successful: Converted model using TF HF")
            except:
                logging.info("Failed: Converted model using TF HF")
                pass
        if convert_success is False and convert_pt_fn:
            try:
                convert_pt_fn(hf_model_name)
                logging.info("Successful: Converted model using PT HF")
                convert_success = True
            except:
                logging.info("Failed to convert model from huggingface")
                pass

        if convert_success:
            model.save_checkpoint(str(self.model_path), overwrite=True)
            logging.info(
                "Successful: Asserted and Converted `{}` from HF and saved it in cache folder {}".format(
                    hf_model_name, str(self.model_path)
                )
            )
        else:
            model.save_checkpoint(str(self.model_path), overwrite=True)
            logging.info(
                "Saved model in cache folder with randomly initialized values  {}".format(str(self.model_path))
            )
