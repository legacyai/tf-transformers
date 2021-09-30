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
"""Run MNLI Mismatched validation"""

import os
from mnli import get_classification_model

from tf_transformers.callbacks.metrics import SklearnMetricCallback
from tf_transformers.models import Classification_Model


def run_mnli_mismatched_evaluation(model_dir, number_of_checkpoints, return_all_layer_outputs):
    """MNLI Mismatched evaluation"""

    num_classes = 2
    is_training = False
    use_dropout = False
    model = get_classification_model(num_classes, return_all_layer_outputs, is_training, use_dropout)()

    for i in range(1, number_of_checkpoints + 1):
        # Load checkpoint
        ckpt_path = os.path.join("/tmp/models/mnli", "ckpt-{}".format(i))

        model.load_checkpoint(ckpt_path)


if __name__ == "__main__":
    model_dir = '/tmp/models/mnli'
    number_of_checkpoints = 1
    return_all_layer_outputs = True
    run_mnli_mismatched_evaluation(model_dir, number_of_checkpoints, return_all_layer_outputs)
