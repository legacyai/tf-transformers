"""TFTBechmark scripts"""
import shutil
import tempfile
import time

import tensorflow as tf
import tqdm
from datasets import load_dataset
from transformers import AlbertTokenizerFast

from tf_transformers.core import ClassificationChainer
from tf_transformers.models import AlbertModel as Model
from tf_transformers.models import AlbertTokenizerTFText, Classification_Model

_ALLOWED_DECODER_TYPES = ["keras_model", "saved_model", "saved_model_tftext"]


class TftBenchmark:
    def __init__(self, cfg):

        self.cfg = cfg

        # Check compatible model type
        self.model_type = cfg.benchmark.model.type
        if self.model_type not in _ALLOWED_DECODER_TYPES:
            raise ValueError("Unknow model type {} defined".format(self.model_type))

        self.model_name = cfg.benchmark.model.name
        max_length = cfg.benchmark.data.max_length

        self.tokenizer = AlbertTokenizerFast.from_pretrained(self.model_name)
        self.tokenizer_tftext = AlbertTokenizerTFText.from_pretrained(
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

        # Add summarize: with text
        self.dataset = dataset

        dataset = dataset.map(
            lambda e: tokenizer(e["text"], truncation=True, padding=True, max_length=max_length),
            batched=True,
        )
        dataset.set_format(type="tensorflow", columns=["input_ids"])
        features = {
            x: tf.cast(dataset[x], dtype=tf.int32).to_tensor(default_value=0, shape=[None, max_length])
            for x in ["input_ids"]
        }
        features['input_mask'] = tf.ones_like(features['input_ids'])
        features['input_type_ids'] = tf.zeros_like(features['input_ids'])

        tfdataset = tf.data.Dataset.from_tensor_slices((features)).batch(batch_size)
        # Convert alldataset to a list for not including that latency while measuring model
        # performance
        # (batch_dataset, batch_size, seq_length)
        batched_datasets = [
            (batch_dataset, batch_dataset['input_ids'].shape[0], batch_dataset['input_ids'].shape[1])
            for batch_dataset in tfdataset
        ]
        return batched_datasets

    def _load_keras_model(self):
        """Load using TextDecoder KerasModel"""

        def decoder_fn(model):
            def _decoder_fn(inputs):
                return model(inputs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name)
        model = Classification_Model(model, num_classes=2)
        model = model.get_model()

        return decoder_fn(model)

    def _load_saved_model(self):
        """Load using TextDecoder saved_model"""

        def decoder_fn(loaded):
            model = loaded.signatures['serving_default']

            def _decoder_fn(inputs):
                return model(**inputs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name
        model = Model.from_pretrained(model_name=model_name)
        model = Classification_Model(model, num_classes=2)
        model = model.get_model()

        # Save as saved_model
        model.save_serialized(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        loaded = tf.saved_model.load(self.temp_dir)

        return decoder_fn(loaded)

    def _load_saved_model_tftext(self):
        """Load using TextDecoder saved_model"""

        def decoder_fn(loaded):
            model = loaded.signatures['serving_default']

            def _decoder_fn(inputs):
                return model(**inputs)

            return _decoder_fn

        model_name = self.cfg.benchmark.model.name

        tokenizer = self.tokenizer_tftext.get_model()
        model = Model.from_pretrained(model_name=model_name)
        model = Classification_Model(model, num_classes=2)
        model = model.get_model()

        model_fully_serialized = ClassificationChainer(tokenizer, model)
        model_fully_serialized = model_fully_serialized.get_model()

        # Save as saved_model
        model_fully_serialized.save_serialized(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        del model_fully_serialized
        loaded = tf.saved_model.load(self.temp_dir)

        return decoder_fn(loaded)

    def load_model_decoder_fn(self):
        """Load Model"""
        if self.model_type == "keras_model":
            decoder_fn = self._load_keras_model()

        if self.model_type == "saved_model":
            decoder_fn = self._load_saved_model()

        if self.model_type == "saved_model_tftext":
            decoder_fn = self._load_saved_model_tftext()

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
        if self.model_type == 'saved_model_tftext':

            all_documents = []
            for item in self.dataset:
                all_documents.append(item['text'])

            text_dataset = tf.data.Dataset.from_tensor_slices({'text': all_documents}).batch(
                self.cfg.benchmark.data.batch_size, drop_remainder=False
            )
            # Sample batch (to avoid first time compilation time)
            for _batch_inputs in tqdm.tqdm(text_dataset, unit="batch "):
                break
            sample_batch_inputs = _batch_inputs
            outputs = decoder_fn(sample_batch_inputs)  # noqa

            slines = 0
            start_time = time.time()
            for batch_inputs in tqdm.tqdm(text_dataset, unit="batch "):
                outputs = decoder_fn(batch_inputs)  # noqa
                batch_size = batch_inputs['text'].shape[0]
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
