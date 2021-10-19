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

import datasets
from model import get_tokenizer


def run_mnli_mismatched_evaluation(
    model, model_dir, write_tfrecord, read_tfrecord, metric_callback, number_of_checkpoints, max_seq_length
):
    """MNLI Mismatched evaluation"""

    data = datasets.load_dataset("glue", 'mnli')

    # Validation matched
    tokenizer = get_tokenizer()
    tfrecord_dir = "/tmp/glue/mnli_mismatched/"
    take_sample = False
    eval_batch_size = 32

    write_tfrecord(
        data["validation_mismatched"],
        max_seq_length,
        tokenizer,
        tfrecord_dir,
        mode="eval",
        take_sample=take_sample,
        verbose=1000,
    )

    # Read TFRecords Validation
    eval_tfrecord_dir = os.path.join(tfrecord_dir, "eval")
    eval_dataset, total_eval_examples = read_tfrecord(
        eval_tfrecord_dir, eval_batch_size, shuffle=False, drop_remainder=False
    )

    results_per_epoch = []
    for i in range(1, number_of_checkpoints + 1):
        # Load checkpoint
        ckpt_path = os.path.join(model_dir, "ckpt-{}".format(i))

        model.load_checkpoint(checkpoint_path=ckpt_path)

        result = metric_callback({"model": model, "validation_dataset": eval_dataset})
        results_per_epoch.append(result)

    return results_per_epoch
