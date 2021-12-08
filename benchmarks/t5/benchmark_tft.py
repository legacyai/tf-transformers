"""TFTBechmark scripts"""
import contextlib
import shutil
import tempfile
import time

import tensorflow as tf
import tqdm
from datasets import load_dataset
from transformers import T5TokenizerFast

from tf_transformers.core import TextGenerationChainer
from tf_transformers.models import T5Model as Model
from tf_transformers.models import T5TokenizerTFText
from tf_transformers.text import TextDecoder, TextDecoderSerializable

_ALLOWED_DECODER_TYPES = [
    "textdecoder_keras_model",
    "textdecoder_saved_model",
    "textdecoder_serializable",  # This uses tf.while_loop
    "textdecoder_serializable_tftext",  # This uses while loop (tf.keras.Model) + tftext tokenizer
]


class TftBenchmark:
    def __init__(self, cfg):

        self.cfg = cfg

        # Check compatible model type
        self.model_type = cfg.benchmark.model.type
        if self.model_type not in _ALLOWED_DECODER_TYPES:
            raise ValueError("Unknow model type {} defined".format(self.model_type))

        self.model_name = cfg.benchmark.model.name
        max_length = cfg.benchmark.data.max_length

        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
        self.tokenizer_tftext = T5TokenizerTFText.from_pretrained(
            self.model_name, add_special_tokens=True, max_length=max_length, dynamic_padding=True, truncate=True
        )

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

        dataset = load_dataset(dataset_name, split="test")

        if take_sample:
            dataset = dataset.select(range(50))

        self.dataset = dataset

        dataset = dataset.map(
            lambda e: tokenizer(e["document"], truncation=True, padding=True, max_length=max_length),
            batched=True,
        )
        dataset = dataset.map(
            lambda x: {'encoder_input_ids': x['input_ids'], 'encoder_input_mask': [1] * len(x['input_ids'])}
        )
        dataset.set_format(type="tensorflow", columns=["encoder_input_ids", "encoder_input_mask"])
        features = {
            x: tf.cast(dataset[x], dtype=tf.int32).to_tensor(default_value=0, shape=[None, max_length])
            for x in ["encoder_input_ids", "encoder_input_mask"]
        }
        tfdataset = tf.data.Dataset.from_tensor_slices((features)).batch(batch_size)
        # Convert alldataset to a list for not including that latency while measuring model
        # performance
        # (batch_dataset, batch_size)
        batched_datasets = [(batch_dataset, batch_dataset['encoder_input_ids'].shape[0]) for batch_dataset in tfdataset]
        return batched_datasets

    def _load_textdecoder_keras_model(self):
        """Load using TextDecoder KerasModel"""

        def decoder_fn(text_generation_kwargs):
            def _decoder_fn(inputs):
                return decoder.decode(inputs, **text_generation_kwargs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)

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
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)

        # Save as saved_model
        model.save_as_serialize_module(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        loaded = tf.saved_model.load(self.temp_dir)

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
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
        # Make decoder model
        text_generation_kwargs = self.cfg.benchmark.text_generation
        decoder = TextDecoderSerializable(model=model, **text_generation_kwargs)

        # Save as saved_model
        decoder.save_serialized(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        del decoder
        # It should be a part of the class (self.loaded)
        self.loaded = tf.saved_model.load(self.temp_dir)

        return decoder_fn()

    def _load_textdecoder_serializable_tftext(self):
        """Load using TextDecoder Serializable tf.while_loop + tftext"""

        def decoder_fn():
            model = self.loaded.signatures['serving_default']

            def _decoder_fn(inputs):
                return model(**inputs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
        # Make decoder model
        text_generation_kwargs = self.cfg.benchmark.text_generation
        decoder = TextDecoderSerializable(model=model, **text_generation_kwargs)

        model_fully_serialized = TextGenerationChainer(self.tokenizer_tftext.get_model(), decoder)
        model_fully_serialized = model_fully_serialized.get_model()
        # Save as saved_model
        model_fully_serialized.save_serialized(self.temp_dir, overwrite=True)

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
            model = Model.from_pretrained(model_name=model_name)
            # Load Auto Regressive Version
            model = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
            # Make decoder model

            text_generation_kwargs = self.cfg.benchmark.text_generation
            decoder = TextDecoderSerializable(model=model, **text_generation_kwargs)

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

        from tf_transformers.text import TextDecoderModel

        model_name = self.cfg.benchmark.model.name
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
        # Make decoder model

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

        from tf_transformers.text import TextDecoderModel

        model_name = self.cfg.benchmark.model.name
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
        # Make decoder model

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

        # This requires text as tensor slices, not features using T5Tokenizer
        # from HuggingFace
        if self.model_type == 'textdecoder_serializable_tftext':

            all_documents = []
            for item in self.dataset:
                all_documents.append(item['document'])

            text_dataset = tf.data.Dataset.from_tensor_slices({'text': all_documents}).batch(
                self.cfg.benchmark.data.batch_size, drop_remainder=True
            )
            # Sample batch (to avoid first time compilation time)
            sample_batch_inputs, _ = text_dataset.take(1)
            outputs = decoder_fn(sample_batch_inputs)

            slines = 0
            start_time = time.time()
            for batch_inputs in tqdm.tqdm(text_dataset, unit="batch "):
                outputs = decoder_fn(batch_inputs)  # noqa
                batch_size = batch_inputs['encoder_input_ids'].shape[0]
                slines += batch_size
            end_time = time.time()
            shutil.rmtree(self.temp_dir)

            time_taken = end_time - start_time
            samples_per_second = slines / time_taken
        else:
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

        return {"model_type": self.model_type, "time_taken": time_taken, "samples_per_second": samples_per_second}
