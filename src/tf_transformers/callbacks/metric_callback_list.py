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
"""This will return tf.keras.metric object based on name"""

import tensorflow as tf

_ALL_METRIC_NAMES = ['binary_accuracy']


def show_available_metric_names():
    print(_ALL_METRIC_NAMES)


def get_callback(metric_name: str):
    """Return tf.keras.metric with a name, for callback"""
    metric_name = metric_name.lower().strip()

    if metric_name not in _ALL_METRIC_NAMES:
        raise ValueError("{} not present in {}".format(metric_name, _ALL_METRIC_NAMES))

    if metric_name == "binary_accuracy":
        return tf.keras.metrics.BinaryAccuracy, metric_name
