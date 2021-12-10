"""HFBechmark scripts"""
import glob
import os
import shutil
import tempfile
import time

import tensorflow as tf
import tqdm
from PIL import Image
from transformers import (
    TFViTForImageClassification,
    ViTFeatureExtractor,
    ViTForImageClassification,
)

_ALLOWED_DECODER_TYPES = [
    "tf",
    "pt",
    "jax",
]


def batchify(data, batch_size):
    length = len(data)
    for ndx in range(0, length, batch_size):
        yield data[ndx : min(ndx + batch_size, length)]


class HFBenchmark:
    def __init__(self, cfg):

        self.cfg = cfg

        # Check compatible model type
        self.model_type = cfg.benchmark.model.type
        if self.model_type not in _ALLOWED_DECODER_TYPES:
            raise ValueError("Unknow model type {} defined".format(self.model_type))

        self.model_name = cfg.benchmark.model.name

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        self.data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
        self.all_flower_path = glob.glob(os.path.join(self.data_dir, '*/*'))
        self.temp_dir = tempfile.mkdtemp()

    def _load_tf(self):
        """Load using KerasModel"""

        def classifier_fn(model):
            def _classifier_fn(inputs):
                return model(**inputs)

            return _classifier_fn

        model_name = self.cfg.benchmark.model.name
        model = TFViTForImageClassification.from_pretrained(model_name)
        return classifier_fn(model)

    def _load_pt(self):
        """Load using PTModel"""

        def classifier_fn(model):
            def _classifier_fn(inputs):
                return model(**inputs)

            return _classifier_fn

        import torch

        device = self.cfg.benchmark.task.device
        device = torch.device(device)

        model = ViTForImageClassification.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        return classifier_fn(model)

    def load_model_classifier_fn(self):
        """Load Model"""

        if self.model_type == "tf":
            classifier_fn = self._load_tf()

        if self.model_type == 'pt':
            classifier_fn = self._load_pt()

        return classifier_fn

    def run(self):

        #### Load Decoder function
        classifier_fn = self.load_model_classifier_fn()
        print("Classify function loaded succesfully")

        import gc

        gc.collect()

        slines = 0
        start_time = time.time()

        batch_size = self.cfg.data.batch_size
        for batch_images in tqdm.tqdm(batchify(self.all_flower_path, batch_size), unit="batch "):
            batch_images = [Image.open(file_) for file_ in batch_images]
            batch_size = len(batch_images)
            inputs = self.feature_extractor(images=batch_images, return_tensors="pt")
            outputs = classifier_fn(inputs)  # noqa
            slines += batch_size
        end_time = time.time()
        shutil.rmtree(self.temp_dir)

        time_taken = end_time - start_time
        samples_per_second = slines / time_taken

        return {"model_type": self.model_type, "time_taken": time_taken, "samples_per_second": samples_per_second}
