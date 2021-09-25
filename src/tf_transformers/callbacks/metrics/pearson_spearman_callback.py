# coding=utf-8
# Copyright 2021 TF-Transformers Authors.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple metric callback for some common metrics in Tensorflow 2.0"""

import tensorflow as tf
import tqdm
from absl import logging
from scipy.stats import pearsonr, spearmanr

_ALL_METRIC_NAMES = {'pearsonr': pearsonr, 'spearmanr': spearmanr}


class PearsonSpearmanCallback:
    def __init__(
        self,
        metric_name_list=('pearsonr', 'spearmanr'),
        label_column: str = 'labels',
        prediction_column: str = 'class_logits',
        validation_dataset: tf.data.Dataset = None,
    ) -> None:
        """

        Args:
            label_column (str): the key from data dict.
            y_column (str): the key of model predictions.
            validation_dataset (tf.data.Dataset, optional): Validation dataset
        """
        for metric_name in metric_name_list:
            if metric_name not in _ALL_METRIC_NAMES:
                raise ValueError(
                    "metric {} not found in supported metric list {}".format(metric_name, _ALL_METRIC_NAMES)
                )
        self.metric_name_list = metric_name_list
        self.validation_dataset = validation_dataset
        self.label_column = label_column
        self.prediction_column = prediction_column

    def __call__(self, traininer_kwargs):
        """This is getting called inside the trainer class"""
        logging.info("Callback for {} is in progress . . . . . . . . . .".format(self.metric_name_list))
        # This is non distribute
        validation_dataset = traininer_kwargs['validation_dataset']
        # No validation dataset has been provided
        if validation_dataset is None:
            if self.validation_dataset is None:
                raise ValueError(
                    "No validation dataset has been provided either in the trainer class, \
                                 or when callback is initialized. Please provide a validation dataset"
                )
            else:
                validation_dataset = self.validation_dataset

        def run_dataset_non_distributed(dataset):
            """The step function for one validation step"""

            predicted_scores_full = []
            labels_full = []

            def validate_step(dist_inputs):
                """The computation to run on each device."""
                batch_inputs, batch_labels = dist_inputs
                model_outputs = model(batch_inputs)
                predicted_scores = tf.squeeze(model_outputs[self.prediction_column], axis=1).numpy()

                predicted_scores_full.extend(predicted_scores)
                labels_full.extend(tf.squeeze(batch_labels[self.label_column], axis=1).numpy())

            for dist_inputs in tqdm.tqdm(dataset):
                validate_step(dist_inputs)

            result = {}
            for metric_name in self.metric_name_list:
                metric_obj = _ALL_METRIC_NAMES[metric_name]
                score = metric_obj(predicted_scores_full, labels_full)
                result[metric_name] = score[0]
            return result

        def run_dataset_all_layers_non_distributed(num_hidden_layers, dataset):
            """The step function for one validation step"""

            predictions_per_layer = {i + 1: [] for i in range(num_hidden_layers)}
            labels_full = []

            def validate_step(dist_inputs):
                """The computation to run on each device."""
                batch_inputs, batch_labels = dist_inputs
                model_outputs = model(batch_inputs)[self.prediction_column]
                labels_full.extend(tf.squeeze(batch_labels[self.label_column], axis=1).numpy())

                layer_no = 1
                for per_layer_output in model_outputs:
                    predicted_scores = tf.squeeze(per_layer_output, axis=1).numpy()
                    predictions_per_layer[layer_no].extend(predicted_scores)
                    layer_no += 1

            for dist_inputs in tqdm.tqdm(dataset):
                validate_step(dist_inputs)

            result = {}
            for layer_no, predicted_scores_full in predictions_per_layer.items():
                for metric_name in self.metric_name_list:
                    metric_obj = _ALL_METRIC_NAMES[metric_name]
                    score = metric_obj(predicted_scores_full, labels_full)
                    result['{}_layer_{}'.format(metric_name, str(layer_no))] = score[0]

            return result

        # Model from trainer
        model = traininer_kwargs['model']
        # Strategy
        # strategy = traininer_kwargs['self'].distribution_strategy

        # Determine whether we need to provide metrics for all layers or single layer
        if isinstance(model.output[self.prediction_column], list):
            num_layers = len(model.output[self.prediction_column])
            result = run_dataset_all_layers_non_distributed(num_layers, validation_dataset)
            return result
        else:
            # Single layer metric
            result = run_dataset_non_distributed(validation_dataset)
            return result
