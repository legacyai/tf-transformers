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
"""A simple callback for some common metrics in Tensorflow 2.0"""
import tensorflow as tf
from absl import logging

from tf_transformers.callbacks.metric_callback_list import get_callback


class MetricCallback:
    def __init__(
        self,
        metric_name: str,
        label_column: str = 'labels',
        prediction_column: str = 'class_logits',
        validation_dataset: tf.data.Dataset = None,
    ) -> None:
        """

        Args:
            metric_name (str):  name of the metric
            label_column (str): the key from data dict.
            y_column (str): the key of model predictions.
            validation_dataset (tf.data.Dataset, optional): Validation dataset
        """
        self.validation_dataset = validation_dataset
        self.label_column = label_column
        self.prediction_column = prediction_column

        self.metric_obj, self.metric_name = get_callback(metric_name)

    def __call__(self, traininer_kwargs):
        """This is getting called inside the trainer class"""
        logging.info("Callback for {} is in progress . . . . . . . . . .".format(self.metric_name))
        # This is strategy.experimemtal_distribute_dataset
        validation_dataset_distributed = traininer_kwargs['validation_dataset_distributed']
        # No validation dataset has been provided
        if validation_dataset_distributed is None:
            if self.validation_dataset is None:
                raise ValueError(
                    "No validation dataset has been provided either in the trainer class, \
                                 or when callback is initialized. Please provide a validation dataset"
                )

        @tf.function
        def run_dataset(dataset):
            """The step function for one validation step"""

            def validate_step(dist_inputs):
                """The computation to run on each device."""
                batch_inputs, batch_labels = dist_inputs
                model_outputs = model(batch_inputs)
                predicted_ids = tf.argmax(model_outputs[self.prediction_column], axis=1)
                # metric_dict['sample'](batch_labels, predicted_ids)
                metric.update_state(batch_labels[self.label_column], predicted_ids)

            for dist_inputs in dataset:
                strategy.run(validate_step, args=(dist_inputs,))

        @tf.function
        def run_dataset_all_layers(dataset):
            """The step function for one validation step"""

            def validate_step(dist_inputs):
                """The computation to run on each device."""
                batch_inputs, batch_labels = dist_inputs
                model_outputs = model(batch_inputs)[self.prediction_column]

                layer_no = 1
                for per_layer_output in model_outputs:
                    predicted_ids = tf.argmax(per_layer_output, axis=1)
                    metric_dict[layer_no].update_state(batch_labels[self.label_column], predicted_ids)
                    layer_no += 1

            for dist_inputs in dataset:
                strategy.run(validate_step, args=(dist_inputs,))

        # Model from trainer
        model = traininer_kwargs['model']
        # Strategy
        strategy = traininer_kwargs['self'].distribution_strategy
        # Determine whether we need to provide metrics for all layers or single layer
        if isinstance(model.output[self.prediction_column], list):
            num_layers = len(model.output[self.prediction_column])
            metric_dict = {
                i + 1: self.metric_obj(name=self.metric_name + '_{}'.format(i + 1)) for i in range(num_layers)
            }

            if validation_dataset_distributed:
                run_dataset_all_layers(validation_dataset_distributed)
            elif self.validation_dataset:
                run_dataset_all_layers(self.validation_dataset)

            result = {}
            for i in range(1, num_layers + 1):
                result['{}_layer_{}'.format(self.metric_name, str(i))] = metric_dict[i].result().numpy()
                metric_dict[i].reset_states()
            return result
        else:
            # Single layer metric
            metric = self.metric_obj(name=self.metric_name)
            if validation_dataset_distributed:
                run_dataset(validation_dataset_distributed)
            elif self.validation_dataset:
                run_dataset(self.validation_dataset)

            result = metric.result()
            metric.reset_states()
            return {self.metric_name: result.numpy()}
