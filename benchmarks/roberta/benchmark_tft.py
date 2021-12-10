"""TFTBechmark scripts"""
import shutil
import tempfile
import time

import tensorflow as tf
import tqdm
from datasets import load_dataset
from transformers import RobertaTokenizerFast

from tf_transformers.models import Classification_Model
from tf_transformers.models import RobertaModel as Model

_ALLOWED_DECODER_TYPES = ["keras_model", "saved_model"]


class TftBenchmark:
    def __init__(self, cfg):

        self.cfg = cfg

        # Check compatible model type
        self.model_type = cfg.benchmark.model.type
        if self.model_type not in _ALLOWED_DECODER_TYPES:
            raise ValueError("Unknow model type {} defined".format(self.model_type))

        self.model_name = cfg.benchmark.model.name

        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
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
        batched_datasets = [(batch_dataset, batch_dataset['input_ids'].shape[0]) for batch_dataset in tfdataset]
        return batched_datasets

    def _load_keras_model(self):
        """Load using TextDecoder KerasModel"""

        def classifier_fn(model):
            def _classifier_fn(inputs):
                return model(inputs)

            return _classifier_fn

        model_name = self.cfg.benchmark.model.name
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name)
        model = Classification_Model(model, num_classes=2)
        model = model.get_model()

        return classifier_fn(model)

    def _load_saved_model(self):
        """Load using TextDecoder saved_model"""

        def classifier_fn():
            model = self.loaded.signatures['serving_default']

            def _classifier_fn(inputs):
                return model(**inputs)

            return _classifier_fn

        model_name = self.cfg.benchmark.model.name
        model = Model.from_pretrained(model_name=model_name)
        model = Classification_Model(model, num_classes=2)
        model = model.get_model()

        # Save as saved_model
        model.save_serialized(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        self.loaded = tf.saved_model.load(self.temp_dir)

        return classifier_fn()

    def load_model_classifier_fn(self):
        """Load Model"""
        if self.model_type == "keras_model":
            classifier_fn = self._load_keras_model()

        if self.model_type == "saved_model":
            classifier_fn = self._load_saved_model()

        return classifier_fn

    def run(self):

        #### Load Decoder function
        classifier_fn = self.load_model_classifier_fn()
        print("Decoder function loaded succesfully")

        #### Load dataset
        batched_datasets = self.load_and_batch_dataset()
        print("Dataset loaded succesfully")

        import gc

        gc.collect()

        #### Run classifier function

        # Sample batch (to avoid first time compilation time)
        sample_batch_inputs, _ = batched_datasets[0]
        outputs = classifier_fn(sample_batch_inputs)

        slines = 0
        start_time = time.time()
        for (batch_inputs, batch_size) in tqdm.tqdm(batched_datasets, unit="batch "):
            outputs = classifier_fn(batch_inputs)  # noqa
            slines += batch_size
        end_time = time.time()
        shutil.rmtree(self.temp_dir)

        time_taken = end_time - start_time
        samples_per_second = slines / time_taken

        return {"model_type": self.model_type, "time_taken": time_taken, "samples_per_second": samples_per_second}
