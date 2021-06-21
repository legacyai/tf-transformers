"""TFTBechmark scripts"""
import contextlib
import shutil
import tempfile
import time

import tensorflow as tf
import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from tf_transformers.models import GPT2Model as Model

_ALLOWED_DECODER_TYPES = [
    "textdecoder_keras_model",
    "textdecoder_saved_model",
    "textdecoder_serializable",  # This uses tf.while_loop
    "textdecoder_model",  # This uses while loop (tf.keras.Model)
    "textdecoder_model_serializable",
]  # This uses while loop


class TftBenchmark:
    def __init__(self, cfg):

        self.cfg = cfg

        # Check compatible model type
        self.model_type = cfg.benchmark.model.type
        if self.model_type not in _ALLOWED_DECODER_TYPES:
            raise ValueError("Unknow model type {} defined".format(self.model_type))

        self.model_name = cfg.benchmark.model.name

        # Hardcode gpt-medium (gpt2 is not working some issues)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token_id

        self.temp_dir = tempfile.mkdtemp()

    def load_and_batch_dataset(self):
        """Load TF dataset"""
        cfg = self.cfg
        tokenizer = self.tokenizer

        # Load from hydra config
        dataset_name = cfg.benchmark.data.name
        take_sample = cfg.benchmark.data.take_sample
        batch_size = cfg.benchmark.data.batch_size
        max_length = cfg.benchmark.data.max_length

        dataset = load_dataset(dataset_name, "3.0.0", split="test")
        if take_sample:
            dataset = dataset.select(range(50))

        dataset = dataset.map(
            lambda e: tokenizer(e["article"], truncation=True, padding=True, max_length=max_length),
            batched=True,
        )
        dataset.set_format(type="tensorflow", columns=["input_ids"])
        features = {
            x: tf.cast(dataset[x], dtype=tf.int32).to_tensor(default_value=0, shape=[None, max_length])
            for x in ["input_ids"]
        }
        tfdataset = tf.data.Dataset.from_tensor_slices((features)).batch(batch_size)
        # Convert alldataset to a list for not including that latency while measuring model
        # performance
        # (batch_dataset, batch_size)
        batched_datasets = [(batch_dataset, batch_dataset['input_ids'].shape[0]) for batch_dataset in tfdataset]
        return batched_datasets

    def _load_textdecoder_keras_model(self):
        """Load using TextDecoder KerasModel"""

        def decoder_fn(text_generation_kwargs):
            def _decoder_fn(inputs):
                return decoder.decode(inputs, **text_generation_kwargs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        model, model_config = Model.from_pretrained(model_name=model_name)
        # Load Auto Regressive Version
        model, model_config = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)

        from tf_transformers.text import TextDecoder

        decoder = TextDecoder(model=model)

        text_generation_kwargs = self.cfg.benchmark.text_generation
        return decoder_fn(text_generation_kwargs)

    def _load_textdecoder_saved_model(self):
        """Load using TextDecoder saved_model"""

        def decoder_fn(text_generation_kwargs):
            def _decoder_fn(inputs):
                return decoder.decode(inputs, **text_generation_kwargs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        model, model_config = Model.from_pretrained(model_name=model_name)
        # Load Auto Regressive Version
        model, model_config = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)

        # Save as saved_model
        model.save_as_serialize_module(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        loaded = tf.saved_model.load(self.temp_dir)

        from tf_transformers.text import TextDecoder

        decoder = TextDecoder(model=loaded)

        text_generation_kwargs = self.cfg.benchmark.text_generation
        return decoder_fn(text_generation_kwargs)

    def _load_textdecoder_serializable(self):
        """Load using TextDecoder Serializable tf.while_loop"""

        def decoder_fn():
            model = self.loaded.signatures['serving_default']

            def _decoder_fn(inputs):
                return model(**inputs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        model, model_config = Model.from_pretrained(model_name=model_name)
        # Load Auto Regressive Version
        model, model_config = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
        # Make decoder model
        from tf_transformers.text import TextDecoderSerializable

        text_generation_kwargs = self.cfg.benchmark.text_generation
        decoder = TextDecoderSerializable(model=model, **text_generation_kwargs)
        decoder = decoder.get_model()

        # Save as saved_model
        decoder.save_serialized(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        del decoder
        # It should be a part of the class (self.loaded)
        self.loaded = tf.saved_model.load(self.temp_dir)

        return decoder_fn()

    def _load_textdecoder_serializable_grappler(self):
        """Load using TextDecoder Serializable tf.while_loop"""

        @contextlib.contextmanager
        def options(options):
            old_opts = tf.config.optimizer.get_experimental_options()
            tf.config.optimizer.set_experimental_options(options)
            try:
                yield
            finally:
                tf.config.optimizer.set_experimental_options(old_opts)

        with options(
            {
                "constant_folding": True,
                "shape_optimization": True,
                "disable_model_pruning": False,
                "arithmetic_optimization": True,
                "function_optimization": True,
                "remapping": True,
                "dependency_optimization": True,
                "loop_optimization": True,
                "scoped_allocator_optimization": True,
            }
        ):

            def decoder_fn():
                model = self.loaded.signatures['serving_default']

                def _decoder_fn(inputs):
                    return model(**inputs)

                return _decoder_fn

            model_name = self.cfg.benchmark.model.name
            model, model_config = Model.from_pretrained(model_name=model_name)
            # Load Auto Regressive Version
            model, model_config = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
            # Make decoder model
            from tf_transformers.text import TextDecoderSerializable

            text_generation_kwargs = self.cfg.benchmark.text_generation
            decoder = TextDecoderSerializable(model=model, **text_generation_kwargs)
            decoder = decoder.get_model()

            # Save as saved_model
            decoder.save_serialized(self.temp_dir, overwrite=True)

            # Load as saved_model
            del model
            del decoder
            # It should be a part of the class (self.loaded)
            self.loaded = tf.saved_model.load(self.temp_dir)

        return decoder_fn()

    def _load_textdecoder_model_serializable(self):
        """Load using TextDecoder Serializable while loop"""

        def decoder_fn():
            model = self.loaded.signatures['serving_default']

            def _decoder_fn(inputs):
                return model(**inputs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        model, model_config = Model.from_pretrained(model_name=model_name)
        # Load Auto Regressive Version
        model, model_config = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
        # Make decoder model
        from tf_transformers.text import TextDecoderModel

        text_generation_kwargs = self.cfg.benchmark.text_generation
        decoder = TextDecoderModel(model=model, **text_generation_kwargs)
        decoder = decoder.get_model()

        # Save as saved_model
        decoder.save_serialized(self.temp_dir, overwrite=True)

        # delete model
        del model
        del decoder

        # It should be a part of the class (self.loaded)
        self.loaded = tf.saved_model.load(self.temp_dir)

        return decoder_fn()

    def _load_textdecoder_model(self):
        """Load using TextDecoder while loop"""

        def decoder_fn(model):
            def _decoder_fn(inputs):
                return model(inputs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        model, model_config = Model.from_pretrained(model_name=model_name)
        # Load Auto Regressive Version
        model, model_config = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
        # Make decoder model
        from tf_transformers.text import TextDecoderModel

        text_generation_kwargs = self.cfg.benchmark.text_generation
        decoder = TextDecoderModel(model=model, **text_generation_kwargs)
        decoder = decoder.get_model()

        # delete model
        del model

        return decoder_fn(decoder)

    def load_model_decoder_fn(self):
        """Load Model"""

        if self.model_type == "textdecoder_keras_model":
            decoder_fn = self._load_textdecoder_keras_model()

        if self.model_type == "textdecoder_saved_model":
            decoder_fn = self._load_textdecoder_saved_model()

        if self.model_type == "textdecoder_serializable":
            decoder_fn = self._load_textdecoder_serializable()

        if self.model_type == "textdecoder_model":
            decoder_fn = self._load_textdecoder_model()

        if self.model_type == "textdecoder_model_serializable":
            decoder_fn = self._load_textdecoder_model_serializable()

        if self.model_type == 'grappler':
            decoder_fn = self._load_textdecoder_serializable_grappler()

        return decoder_fn

    def run(self):

        #### Load Decoder function
        decoder_fn = self.load_model_decoder_fn()
        print("Decoder function loaded succesfully")

        #### Load dataset
        batched_datasets = self.load_and_batch_dataset()
        print("Dataset loaded succesfully")

        import gc

        gc.collect()

        #### Run decoder function

        # Sample batch (to avoid first time compilation time)
        sample_batch_inputs, _ = batched_datasets[0]
        outputs = decoder_fn(sample_batch_inputs)

        slines = 0
        start_time = time.time()
        for (batch_inputs, batch_size) in tqdm.tqdm(batched_datasets, unit="batch "):
            outputs = decoder_fn(batch_inputs)  # noqa
            slines += batch_size
        end_time = time.time()
        shutil.rmtree(self.temp_dir)

        time_taken = end_time - start_time
        samples_per_second = slines / time_taken

        return {"time_taken": time_taken, "samples_per_second": samples_per_second}
