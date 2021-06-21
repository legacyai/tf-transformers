"""TFTBechmark scripts"""
import shutil
import tempfile
import time

import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast

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

        # Hardcode gpt-medium (gpt2 is not working some issues)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token_id

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

        dataset = load_dataset(dataset_name, "3.0.0", split="test")
        if take_sample:
            dataset = dataset.select(range(50))

        dataset = dataset.map(
            lambda e: tokenizer(e["article"], truncation=True, padding=True, max_length=max_length),
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

    def load_and_batch_dataset(self):
        """Load Dataset based on model type"""
        if self.model_type == "tf":
            dataset = self._load_dataset_tf()

        if self.model_type == 'pt':
            dataset = self._load_dataset_pt()

        return dataset

    def _load_tf(self):
        """Load using KerasModel"""
        import tensorflow as tf

        def decoder_fn(model, text_generation_kwargs):
            text_generation_kwargs = dict(text_generation_kwargs)
            del text_generation_kwargs['max_length']  # we will pass it from inputs

            def _decoder_fn(inputs, max_length):
                return model.generate(**inputs, max_length=max_length, **text_generation_kwargs)

            return _decoder_fn

        from transformers import GPT2Config as Config
        from transformers import TFGPT2LMHeadModel as Model

        # model_name = self.cfg.benchmark.model.name
        # model = Model.from_pretrained(model_name=model_name) # somehow link is broken

        configuration = Config()
        model = Model(configuration)
        # Dummy initialize
        dummy = model(input_ids=tf.constant([[1, 2]]))  # noqa

        text_generation_kwargs = self.cfg.benchmark.text_generation
        return decoder_fn(model, text_generation_kwargs)

    def _load_pt(self):
        """Load using KerasModel"""

        def decoder_fn(model, text_generation_kwargs):
            text_generation_kwargs = dict(text_generation_kwargs)
            del text_generation_kwargs['max_length']  # we will pass it from inputs

            def _decoder_fn(inputs, max_length):
                return model.generate(**inputs, max_length=max_length, **text_generation_kwargs)

            return _decoder_fn

        import torch
        from transformers import GPT2Config as Config
        from transformers import GPT2LMHeadModel as Model

        device = self.cfg.benchmark.task.device
        device = torch.device(device)

        # model_name = self.cfg.benchmark.model.name
        # model = Model.from_pretrained(model_name=model_name) # somehow link is broken

        configuration = Config()
        model = Model(configuration)
        model.to(device)
        model.eval()

        text_generation_kwargs = self.cfg.benchmark.text_generation
        return decoder_fn(model, text_generation_kwargs)

    def load_model_decoder_fn(self):
        """Load Model"""

        if self.model_type == "tf":
            decoder_fn = self._load_tf()

        if self.model_type == 'pt':
            decoder_fn = self._load_pt()

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
        max_length = self.cfg.benchmark.text_generation.max_length
        # Sample batch (to avoid first time compilation time)
        sample_batch_inputs, _, seq_length = batched_datasets[0]
        outputs = decoder_fn(sample_batch_inputs, seq_length + max_length)

        slines = 0
        start_time = time.time()
        for (batch_inputs, batch_size, seq_length) in tqdm.tqdm(batched_datasets, unit="batch "):
            # HF max_length needs seq_length + max_length
            outputs = decoder_fn(batch_inputs, seq_length + max_length)  # noqa
            slines += batch_size
        end_time = time.time()
        shutil.rmtree(self.temp_dir)

        time_taken = end_time - start_time
        samples_per_second = slines / time_taken

        return {"time_taken": time_taken, "samples_per_second": samples_per_second}
