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
# The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual
# entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether
# the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).
# The premise sentences are gathered from ten different sources, including transcribed speech, fiction,
# and government reports. The authors of the benchmark use the standard test set, for which they obtained
# private labels from the RTE authors, and evaluate on both the matched (in-domain) and mismatched
# (cross-domain) section. They also uses and recommend the SNLI corpus as 550k examples of auxiliary training data.
"""MNLI in Tensorflow 2.0
Task: 3 class Softmax Classification.
"""

import glob
import json
import os
import tempfile

import datasets
import hydra
from absl import logging
from hydra import compose
from model import get_model, get_optimizer, get_tokenizer, get_trainer
from omegaconf import DictConfig

from tf_transformers.callbacks.metrics import SklearnMetricCallback
from tf_transformers.data import TFReader, TFWriter
from tf_transformers.losses.loss_wrapper import get_1d_classification_loss
from tf_transformers.models import Classification_Model

logging.set_verbosity("INFO")

NUM_LAYERS = 12  # Number of hidden_layers for the Model


def write_tfrecord(
    data, max_seq_length: int, tokenizer, tfrecord_dir: str, mode: str, take_sample=False, verbose=10000
):
    """Write TFRecords"""
    if mode not in ["train", "eval"]:
        raise ValueError("Inavlid mode `{}` specified. Available mode is ['train', 'eval']".format(mode))

    def get_tfrecord_example(data):

        for f in data:
            input_ids_s1 = (
                [tokenizer.cls_token]
                + tokenizer.tokenize(f['hypothesis'])[: max_seq_length - 2]
                + [tokenizer.sep_token]
            )  # -2 to add CLS and SEP
            input_ids_s1 = tokenizer.convert_tokens_to_ids(input_ids_s1)
            input_type_ids_s1 = [0] * len(input_ids_s1)  # 0 for s1

            input_ids_s2 = tokenizer.tokenize(f['premise'])[: max_seq_length - 1] + [
                tokenizer.sep_token
            ]  # -1 to add SEP
            input_ids_s2 = tokenizer.convert_tokens_to_ids(input_ids_s2)
            input_type_ids_s2 = [1] * len(input_ids_s2)  # 1 for s2

            # concatanate two sentences
            input_ids = input_ids_s1 + input_ids_s2
            input_type_ids = input_type_ids_s1 + input_type_ids_s2
            input_mask = [1] * len(input_ids)  # 1 for s2

            result = {}
            result['input_ids'] = input_ids
            result['input_mask'] = input_mask
            result['input_type_ids'] = input_type_ids

            result['labels'] = f['label']
            yield result

    schema = {
        "input_ids": ("var_len", "int"),
        "input_mask": ("var_len", "int"),
        "input_type_ids": ("var_len", "int"),
        "labels": ("var_len", "int"),
    }

    if mode == "train":
        # Write tf records
        train_data_dir = os.path.join(tfrecord_dir, "train")
        # Create a temp dir
        if os.path.exists(train_data_dir):
            logging.info("TFrecords exists in the directory {}".format(train_data_dir))
            return True
        tfwriter = TFWriter(schema=schema, model_dir=train_data_dir, tag='train', verbose_counter=verbose)
        data_train = data
        # Take sample
        if take_sample:
            data_train = data_train.select(range(500))

        tfwriter.process(parse_fn=get_tfrecord_example(data_train))
    if mode == "eval":
        # Write tfrecords
        eval_data_dir = os.path.join(tfrecord_dir, "eval")
        # Create a temp dir
        if os.path.exists(eval_data_dir):
            logging.info("TFrecords exists in the directory {}".format(eval_data_dir))
            return True
        tfwriter = TFWriter(schema=schema, model_dir=eval_data_dir, tag='eval', verbose_counter=verbose)
        data_eval = data
        # Take sample
        if take_sample:
            data_eval = data_eval.select(range(500))
        tfwriter.process(parse_fn=get_tfrecord_example(data_eval))


def read_tfrecord(
    tfrecord_dir: str,
    batch_size: int,
    shuffle: bool = False,
    drop_remainder: bool = False,
    static_padding: bool = False,
    max_seq_length: int = None,
):
    """Read TFRecords"""
    padded_shapes = None
    if static_padding:
        if max_seq_length is None:
            raise ValueError("When `static_padding=Trur`, `max_seq_length` should be set. For eg: 128")
        padded_shapes = {
            'input_ids': [
                max_seq_length,
            ],
            'input_mask': [
                max_seq_length,
            ],
            'input_type_ids': [
                max_seq_length,
            ],
            'labels': [
                None,
            ],
            'labels_mask': [
                None,
            ],
        }
    # Read tfrecord to dataset
    schema = json.load(open("{}/schema.json".format(tfrecord_dir)))
    stats = json.load(open('{}/stats.json'.format(tfrecord_dir)))
    all_files = glob.glob("{}/*.tfrecord".format(tfrecord_dir))
    tf_reader = TFReader(schema=schema, tfrecord_files=all_files)

    x_keys = ['input_ids', 'input_mask', 'input_type_ids']
    y_keys = ['labels']
    dataset = tf_reader.read_record(
        auto_batch=True,
        keys=x_keys,
        padded_shapes=padded_shapes,
        batch_size=batch_size,
        x_keys=x_keys,
        y_keys=y_keys,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
    )
    return dataset, stats['total_records']


def get_classification_model(num_classes: int, return_all_layer_outputs: bool, is_training: bool, use_dropout: bool):
    """Classification Model"""

    def model_fn():
        model = get_model(return_all_layer_outputs, is_training, use_dropout)
        classification_model = Classification_Model(
            model,
            num_classes,
            use_all_layers=return_all_layer_outputs,
            is_training=is_training,
            use_dropout=use_dropout,
        )
        classification_model = classification_model.get_model()
        return classification_model

    return model_fn


@hydra.main(config_path="config")
def run_mnli(cfg: DictConfig):
    logging.info("Run MNLI")
    cfg = compose(config_name="config", overrides=["+glue=mnli"])
    task_name = cfg.glue.task.name
    data_name = cfg.glue.data.name
    max_seq_length = cfg.glue.data.max_seq_length
    take_sample = cfg.data.take_sample
    train_batch_size = cfg.data.train_batch_size

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load data
    data = datasets.load_dataset("glue", data_name)

    # Write TFRecords
    temp_dir = tempfile.gettempdir()
    tfrecord_dir = os.path.join(temp_dir, "glue", "tfrecord", task_name)

    # Train
    write_tfrecord(
        data["train"], max_seq_length, tokenizer, tfrecord_dir, mode="train", take_sample=take_sample, verbose=10000
    )
    # Validation matched
    write_tfrecord(
        data["validation_matched"],
        max_seq_length,
        tokenizer,
        tfrecord_dir,
        mode="eval",
        take_sample=take_sample,
        verbose=1000,
    )

    # Read TFRecords Train
    train_tfrecord_dir = os.path.join(tfrecord_dir, "train")
    train_dataset, total_train_examples = read_tfrecord(
        train_tfrecord_dir, cfg.data.train_batch_size, shuffle=True, drop_remainder=True
    )
    # Read TFRecords Validation
    eval_tfrecord_dir = os.path.join(tfrecord_dir, "eval")
    eval_dataset, total_eval_examples = read_tfrecord(
        eval_tfrecord_dir, cfg.data.eval_batch_size, shuffle=False, drop_remainder=False
    )

    # Load optimizer
    optimizer_fn = get_optimizer(
        cfg.optimizer.learning_rate, total_train_examples, cfg.data.train_batch_size, cfg.trainer.epochs
    )

    # Load trainer
    trainer = get_trainer(
        distribution_strategy=cfg.trainer.strategy, num_gpus=cfg.trainer.num_gpus, tpu_address=cfg.trainer.tpu_address
    )

    # Load model function
    loss_type = cfg.optimizer.loss_type
    return_all_layer_outputs = False
    training_loss_names = None
    validation_loss_names = None
    if loss_type and loss_type == 'joint':
        return_all_layer_outputs = True
        training_loss_names = ['loss_{}'.format(i + 1) for i in range(NUM_LAYERS)]
        validation_loss_names = training_loss_names

    # Load Model
    model_fn = get_classification_model(
        cfg.glue.data.num_classes, return_all_layer_outputs, cfg.model.is_training, cfg.model.use_dropout
    )

    # Load loss fn
    loss_fn = get_1d_classification_loss(loss_type=loss_type)

    # Run
    steps_per_epoch = total_train_examples // train_batch_size
    model_checkpoint_dir = os.path.join(temp_dir, "models", task_name)

    # Callback
    metric_callback = SklearnMetricCallback(metric_name_list=('accuracy_score',))

    history = trainer.run(
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=train_dataset,
        train_loss_fn=loss_fn,
        epochs=cfg.trainer.epochs,
        steps_per_epoch=steps_per_epoch,
        model_checkpoint_dir=model_checkpoint_dir,
        batch_size=train_batch_size,
        training_loss_names=training_loss_names,
        validation_loss_names=validation_loss_names,
        validation_dataset=eval_dataset,
        validation_loss_fn=loss_fn,
        validation_interval_steps=None,
        steps_per_call=1,
        enable_xla=False,
        callbacks=[metric_callback],
        callbacks_interval_steps=None,
        max_number_of_models=10,
        model_save_interval_steps=None,
        repeat_dataset=True,
        latest_checkpoint=None,
    )

    return history
