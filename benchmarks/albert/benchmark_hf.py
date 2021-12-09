"""HFBechmark scripts"""
import shutil
import tempfile
import time

import tqdm
from datasets import load_dataset
from transformers import AlbertTokenizerFast

_ALLOWED_DECODER_TYPES = [
    "tf",
    "pt",
    "jax",
]


class HFBenchmark:
    def __init__(self, cfg):

        self.cfg = cfg

        # Check compatible model type
        self.model_type = cfg.benchmark.model.type
        if self.model_type not in _ALLOWED_DECODER_TYPES:
            raise ValueError("Unknow model type {} defined".format(self.model_type))

        self.model_name = cfg.benchmark.model.name

        self.tokenizer = AlbertTokenizerFast.from_pretrained(self.model_name)

        self.temp_dir = tempfile.mkdtemp()

    def _load_dataset_tf(self):
        """Load TF dataset"""
        import tensorflow as tf

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

        dataset = dataset.map(
            lambda e: tokenizer(e["text"], truncation=True, padding=True, max_length=max_length),
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
        # (batch_dataset, batch_size, seq_length)
        batched_datasets = [
            (batch_dataset, batch_dataset['input_ids'].shape[0], batch_dataset['input_ids'].shape[1])
            for batch_dataset in tfdataset
        ]
        return batched_datasets

    def _load_dataset_pt(self):
        """Load TF dataset"""
        import torch

        cfg = self.cfg
        tokenizer = self.tokenizer
        # Load from hydra config
        dataset_name = cfg.benchmark.data.name
        take_sample = cfg.benchmark.data.take_sample
        batch_size = cfg.benchmark.data.batch_size
        max_length = cfg.benchmark.data.max_length
        device = cfg.benchmark.task.device

        dataset = load_dataset(dataset_name, split="test")

        if take_sample:
            dataset = dataset.select(range(50))

        dataset = dataset.map(
            lambda e: tokenizer(e["text"], truncation=True, padding=True, max_length=max_length),
            batched=True,
        )
        dataset.set_format(type='torch', columns=['input_ids'], device=device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # Convert alldataset to a list for not including that latency while measuring model
        # performance
        # (batch_dataset, batch_size, seq_length)
        batched_datasets = [
            (batch_dataset, batch_dataset['input_ids'].shape[0], batch_dataset['input_ids'].shape[1])
            for batch_dataset in dataloader
        ]
        return batched_datasets

    def _load_dataset_jax(self):
        """Load TF dataset"""
        import jax.numpy as jnp
        import torch

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

        dataset = dataset.map(
            lambda e: tokenizer(e["text"], truncation=True, padding=True, max_length=max_length),
            batched=True,
        )
        dataset.set_format(type='torch', columns=['input_ids'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # Convert alldataset to a list for not including that latency while measuring model
        # performance
        # (batch_dataset, batch_size, seq_length)
        batched_datasets = [
            (
                {"input_ids": jnp.array(batch_dataset["input_ids"].numpy())},
                batch_dataset['input_ids'].shape[0],
                batch_dataset['input_ids'].shape[1],
            )
            for batch_dataset in dataloader
        ]
        return batched_datasets

    def load_and_batch_dataset(self):
        """Load Dataset based on model type"""
        if self.model_type == "tf":
            dataset = self._load_dataset_tf()

        if self.model_type == 'pt':
            dataset = self._load_dataset_pt()

        if self.model_type == 'jax':
            dataset = self._load_dataset_jax()

        return dataset

    def _load_tf(self):
        """Load using KerasModel"""

        def decoder_fn(model):
            def _decoder_fn(inputs):
                return model(**inputs)

            return _decoder_fn

        from transformers import TFAlbertForSequenceClassification as Model

        model_name = self.cfg.benchmark.model.name
        model = Model.from_pretrained(model_name)
        return decoder_fn(model)

    def _load_pt(self):
        """Load using PTModel"""

        def decoder_fn(model):
            def _decoder_fn(inputs):
                return model(**inputs)

            return _decoder_fn

        import torch
        from transformers import AlbertForSequenceClassification as Model

        device = self.cfg.benchmark.task.device
        device = torch.device(device)

        model = Model.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        return decoder_fn(model)

    def _load_jax(self):
        """Load using JaxModel"""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from flax.jax_utils import replicate

        # from flax.training.common_utils import shard
        def my_shard(xs, device_count=1):
            return jax.tree_map(lambda x: x.reshape((device_count, -1) + x.shape[1:]), xs)

        def classify(params, batch, rng):
            output_ids = model(batch["input_ids"], prng_key=rng, params=params)
            return output_ids

        def decoder_fn(p_params, dummy_inputs, rngs):
            p_classify = jax.pmap(classify)
            dummy_outputs = p_classify(p_params, my_shard(dummy_inputs), rngs)  # noqa

            def _decoder_fn(inputs, max_length):
                output_ids = p_classify(p_params, my_shard(inputs), rngs)
                return output_ids

            return _decoder_fn

        from transformers import FlaxAlbertForSequenceClassification as Model

        model_name = self.cfg.benchmark.model.name
        model = Model.from_pretrained(model_name)

        num_devices = 1
        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, num_devices)
        batch_size = self.cfg.benchmark.data.batch_size
        max_length = self.cfg.benchmark.data.max_length
        p_params = replicate(model.params)
        dummy_inputs = jnp.array(np.random.randint(0, 100, size=(batch_size, max_length)), dtype=jnp.int32)
        return decoder_fn(p_params, {"input_ids": dummy_inputs}, rngs)

    def load_model_decoder_fn(self):
        """Load Model"""

        if self.model_type == "tf":
            decoder_fn = self._load_tf()

        if self.model_type == 'pt':
            decoder_fn = self._load_pt()

        if self.model_type == 'jax':
            decoder_fn = self._load_jax()

        return decoder_fn

    def run(self):

        #### Load Decoder function
        decoder_fn = self.load_model_decoder_fn()
        print("Classify function loaded succesfully")

        #### Load dataset
        batched_datasets = self.load_and_batch_dataset()
        print("Dataset loaded succesfully")

        import gc

        gc.collect()

        # Sample batch (to avoid first time compilation time)
        sample_batch_inputs, _, seq_length = batched_datasets[0]  # noqa
        outputs = decoder_fn(sample_batch_inputs)

        slines = 0
        start_time = time.time()
        for (batch_inputs, batch_size, _seq_length) in tqdm.tqdm(batched_datasets, unit="batch "):
            # HF max_length needs seq_length + max_length
            outputs = decoder_fn(batch_inputs)  # noqa
            # print("Outputs", outputs.shape)
            slines += batch_size
        end_time = time.time()
        shutil.rmtree(self.temp_dir)

        time_taken = end_time - start_time
        samples_per_second = slines / time_taken

        return {"model_type": self.model_type, "time_taken": time_taken, "samples_per_second": samples_per_second}
