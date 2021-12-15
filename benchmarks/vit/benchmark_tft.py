"""TFTBechmark scripts"""
import glob
import os
import shutil
import tempfile
import time

import tensorflow as tf
import tqdm
from transformers import ViTFeatureExtractor

from tf_transformers.core import ClassificationChainer
from tf_transformers.models import ViTFeatureExtractorTF
from tf_transformers.models import ViTModel as Model

_ALLOWED_DECODER_TYPES = [
    "keras_model",
    "saved_model",
    "saved_model_tfimage",
    "keras_model_hf_pipeline",
    'saved_model_hf_pipeline',
]


class TftBenchmark:
    def __init__(self, cfg):

        self.cfg = cfg

        # Check compatible model type
        self.model_type = cfg.benchmark.model.type
        if self.model_type not in _ALLOWED_DECODER_TYPES:
            raise ValueError("Unknow model type {} defined".format(self.model_type))

        self.model_name = cfg.benchmark.model.name

        self.feature_extractor = ViTFeatureExtractorTF(img_height=224, img_width=224)
        self.feature_extractor_hf = ViTFeatureExtractor.from_pretrained(self.model_name)

        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        self.data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
        self.all_flower_path = glob.glob(os.path.join(self.data_dir, '*/*'))
        self.temp_dir = tempfile.mkdtemp()

    def _load_keras_model(self):
        """Load using TextDecoder KerasModel"""

        def classifier_fn(model):
            def _classifier_fn(inputs):
                return model(inputs)

            return _classifier_fn

        model_name = self.cfg.benchmark.model.name
        # Load Auto Regressive Version
        model = Model.from_pretrained(model_name=model_name, classification_labels=1000)

        return classifier_fn(model)

    def _load_saved_model(self):
        """Load using TextDecoder saved_model"""

        def classifier_fn():
            model = self.loaded.signatures['serving_default']

            def _classifier_fn(inputs):
                return model(**inputs)

            return _classifier_fn

        model_name = self.cfg.benchmark.model.name
        model = Model.from_pretrained(model_name=model_name, classification_labels=1000)

        # Save as saved_model
        model.save_serialized(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        self.loaded = tf.saved_model.load(self.temp_dir)

        return classifier_fn()

    def _load_saved_model_tfimage(self):
        """Load using TextDecoder saved_model"""

        def classifier_fn():
            model = self.loaded.signatures['serving_default']

            def _classifier_fn(inputs):
                return model(**inputs)

            return _classifier_fn

        model_name = self.cfg.benchmark.model.name

        feature_extractor = self.feature_extractor.get_model()
        model = Model.from_pretrained(model_name=model_name)

        model_fully_serialized = ClassificationChainer(feature_extractor, model)
        model_fully_serialized = model_fully_serialized.get_model()

        # Save as saved_model
        model_fully_serialized.save_serialized(self.temp_dir, overwrite=True)

        # Load as saved_model
        del model
        del model_fully_serialized
        self.loaded = tf.saved_model.load(self.temp_dir)

        return classifier_fn()

    def load_model_classifier_fn(self):
        """Load Model"""
        if self.model_type == "keras_model":
            classifier_fn = self._load_keras_model()

        if self.model_type == "saved_model":
            classifier_fn = self._load_saved_model()

        if self.model_type == "saved_model_tfimage":
            classifier_fn = self._load_saved_model_tfimage()

        if self.model_type == "keras_model_hf_pipeline":
            classifier_fn = self._load_keras_model()

        if self.model_type == "saved_model_hf_pipeline":
            classifier_fn = self._load_saved_model()

        return classifier_fn

    def run(self):

        #### Load Decoder function
        classifier_fn = self.load_model_classifier_fn()
        print("Decoder function loaded succesfully")

        #### Run classifier function

        # This requires text as tensor slices, not features using AlbertTokenizer
        # from HuggingFace
        if self.model_type == 'saved_model_tfimage':

            image_dataset = tf.data.Dataset.from_tensor_slices({'image': self.all_flower_path}).batch(
                self.cfg.benchmark.data.batch_size, drop_remainder=False
            )
            # Sample batch (to avoid first time compilation time)
            for _batch_inputs in tqdm.tqdm(image_dataset, unit="batch "):
                break
            outputs = classifier_fn(_batch_inputs)  # noqa

            slines = 0
            start_time = time.time()
            for batch_inputs in tqdm.tqdm(image_dataset, unit="batch "):
                outputs = classifier_fn(batch_inputs)  # noqa
                batch_size = batch_inputs['image'].shape[0]
                slines += batch_size
            end_time = time.time()
            shutil.rmtree(self.temp_dir)

            time_taken = end_time - start_time
            samples_per_second = slines / time_taken

        elif self.model_type in ['keras_model_hf_pipeline', 'saved_model_hf_pipeline']:

            from PIL import Image

            image_dataset = tf.data.Dataset.from_tensor_slices({'image': self.all_flower_path}).batch(
                self.cfg.benchmark.data.batch_size, drop_remainder=False
            )

            slines = 0
            start_time = time.time()
            for batch_inputs in tqdm.tqdm(image_dataset, unit="batch "):
                batch_inputs = batch_inputs['image'].numpy().tolist()
                batch_images = [Image.open(file_) for file_ in batch_inputs]
                batch_size = len(batch_images)
                batch_inputs = self.feature_extractor_hf(images=batch_images, return_tensors="tf")
                inputs = {}
                inputs['input_pixels'] = tf.transpose(batch_inputs['pixel_values'], [0, 2, 3, 1])
                outputs = classifier_fn(inputs)  # noqa
                slines += batch_size
            end_time = time.time()
            shutil.rmtree(self.temp_dir)

            time_taken = end_time - start_time
            samples_per_second = slines / time_taken

        else:
            image_dataset = tf.data.Dataset.from_tensor_slices({'image': self.all_flower_path}).batch(
                self.cfg.benchmark.data.batch_size, drop_remainder=False
            )
            for _batch_inputs in tqdm.tqdm(image_dataset, unit="batch "):
                _batch_inputs = self.feature_extractor(_batch_inputs)
                break
            outputs = classifier_fn(_batch_inputs)

            slines = 0
            start_time = time.time()
            for batch_inputs in tqdm.tqdm(image_dataset, unit="batch "):
                batch_size = batch_inputs['image'].shape[0]
                batch_inputs = self.feature_extractor(batch_inputs)
                outputs = classifier_fn(batch_inputs)  # noqa
                slines += batch_size
            end_time = time.time()
            shutil.rmtree(self.temp_dir)

            time_taken = end_time - start_time
            samples_per_second = slines / time_taken

        return {"model_type": self.model_type, "time_taken": time_taken, "samples_per_second": samples_per_second}
