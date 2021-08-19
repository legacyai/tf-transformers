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
import os

import tensorflow as tf
import tqdm
from absl import logging

from tf_transformers.core import keras_utils
from tf_transformers.core.distribute_utils import get_distribution_strategy
from tf_transformers.core.performance_utils import (
    get_tf_dtype,
    set_mixed_precision_policy,
)

# logging.get_absl_logger().name = "trainer"


def flat_metric_dict(metric_dict):
    """Flatten the dict"""
    dict_flatten = {}
    dict_flatten['steps'] = list(metric_dict.keys())
    for _key, value in metric_dict.items():
        for sub_key, sub_value in value.items():
            if sub_key not in dict_flatten:
                dict_flatten[sub_key] = [sub_value]
            else:
                dict_flatten[sub_key].append(sub_value)
    return dict_flatten


def save_model_checkpoints(model, overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models, **kwargs):
    # Model checkpoint
    if not overwrite_checkpoint_dir:
        import os

        if os.path.exists(model_checkpoint_dir):
            raise FileExistsError("Model directory exists")

    checkpoint = tf.train.Checkpoint(model=model, **kwargs)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_checkpoint_dir, max_to_keep=max_number_of_models)
    return manager


def get_loss_metric_dict(training_loss_names, validation_loss_names):

    training_loss_dict_metric = {name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in training_loss_names}
    training_loss_dict_metric["learning_rate"] = tf.keras.metrics.Mean(
        "learning_rate", dtype=tf.float32
    )  # We store learning rate here and reset after every global steps

    validation_loss_dict_metric = {}
    if validation_loss_names:
        validation_loss_dict_metric = {
            name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in validation_loss_names
        }

    return training_loss_dict_metric, validation_loss_dict_metric


def get_and_reset_metric_from_dict(metric_dict):
    if not metric_dict:
        return {}
    metric_result = {name: metric.result().numpy() for name, metric in metric_dict.items()}
    for _name, metric in metric_dict.items():
        metric.reset_states()
    return metric_result


def get_tensorboard_writers(model_checkpoint_dir):
    # current_directory = os.getcwd()
    train_log_dir = os.path.join(model_checkpoint_dir, "logs/train")
    test_log_dir = os.path.join(model_checkpoint_dir, "logs/dev")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer


def train_and_eval(
    model,
    optimizer,
    strategy,
    epochs,
    steps_per_epoch,
    steps_per_call,
    train_dataset_iter,
    train_loss_fn,
    GLOBAL_BATCH_SIZE,
    training_loss_dict_metric,
    validation_dataset_distributed,
    validation_loss_fn,
    validation_loss_dict_metric,
    validation_interval_steps,
    callbacks,
    callbacks_interval_steps,
    trainer_kwargs,
    checkpoint_manager,
    model_checkpoint_dir,
    model_save_interval_steps,
):
    def save_model(epoch_end=False):
        if not epoch_end:
            if model_save_interval_steps:
                if global_step % model_save_interval_steps == 0:
                    checkpoint_manager.save()
                    logging.info("Model saved at step {}".format(global_step))
        else:
            checkpoint_manager.save()
            logging.info("Model saved at epoch {}".format(epoch))

    def compute_loss(batch_labels, model_outputs):
        """Loss computation which takes care of loss reduction based on GLOBAL_BATCH_SIZE"""
        per_example_loss = train_loss_fn(batch_labels, model_outputs)
        per_example_loss_averaged = {}
        # Inplace update
        # Avergae loss per global batch size , recommended
        for name, loss in per_example_loss.items():
            per_example_loss_averaged[name] = tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return per_example_loss_averaged

    def compute_loss_valid(batch_labels, model_outputs):
        """Validation Loss computation which takes care of loss reduction based on GLOBAL_BATCH_SIZE"""
        per_example_loss = validation_loss_fn(batch_labels, model_outputs)
        per_example_loss_averaged = {}
        # Inplace update
        # Avergae loss per global batch size , recommended
        for name, loss in per_example_loss.items():
            per_example_loss_averaged[name] = tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return per_example_loss_averaged

    # Train Functions
    @tf.function
    def do_train(iterator):
        """The step function for one training step"""

        def train_step(dist_inputs):
            """The computation to run on each device."""
            batch_inputs, batch_labels = dist_inputs
            with tf.GradientTape() as tape:
                model_outputs = model(batch_inputs)
                loss = compute_loss(batch_labels, model_outputs)

                tf.debugging.check_numerics(loss['loss'], message='Loss value is either NaN or inf')

                # TODO
                # Scales down the loss for gradients to be invariant from replicas.
                # loss = loss / strategy.num_replicas_in_sync
            grads = tape.gradient(loss["loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # training_loss.update_state(loss * strategy.num_replicas_in_sync)
            return loss

        for _ in tf.range(tf.convert_to_tensor(steps_per_call)):
            dist_inputs = next(iterator)
            loss = strategy.run(train_step, args=(dist_inputs,))
            # strategy reduce (SUM) is important
            # If not SUM, final loss might not be a good representative of global batch
            loss = {
                name: strategy.reduce(tf.distribute.ReduceOp.SUM, loss_value, axis=None)
                for name, loss_value in loss.items()
            }
            for name, loss_value in loss.items():
                training_loss = training_loss_dict_metric[name]
                training_loss.update_state(loss_value)
            # Get current learning rate
            current_lr = optimizer._decayed_lr(tf.float32)
            training_loss_dict_metric["learning_rate"].update_state(current_lr)
            # training_result = get_and_reset_metric_from_dict(training_loss_dict_metric)

    # @tf.function(experimental_relax_shapes=True)
    def write_metrics(metric_dict, writer, step):
        # @tf.function
        def _write(step):
            # other model code would go here
            with writer.as_default():
                for name, result in metric_dict.items():
                    tf.summary.scalar(name, result, step=step)

        _write(step)
        writer.flush()

    # do validation
    def do_validation(validation_dataset_distributed):
        """Validation step"""

        @tf.function
        def _validate_step(dist_inputs):

            batch_inputs, batch_labels = dist_inputs
            model_outputs = model(batch_inputs)
            loss = compute_loss_valid(batch_labels, model_outputs)
            return loss

        if not epoch_end:
            if (
                validation_dataset_distributed
                and validation_loss_fn
                and validation_interval_steps
                and (global_step % validation_interval_steps == 0)
            ):
                logging.info("Validation in progress at step {} . . . .".format(global_step))
                with tqdm.tqdm(validation_dataset_distributed, unit=" Val batch ") as val_batches:
                    for dist_inputs in val_batches:
                        loss = strategy.run(_validate_step, args=(dist_inputs,))
                        for name, loss_value in loss.items():
                            loss_value = strategy.reduce(tf.distribute.ReduceOp.SUM, loss_value, axis=None)
                            validation_loss = validation_loss_dict_metric[name]
                            validation_loss.update_state(loss_value)

                validation_result = get_and_reset_metric_from_dict(validation_loss_dict_metric)
                validation_history[global_step] = validation_result
                write_metrics(validation_result, val_summary_writer, global_step)
                logging.info("Validation result at step {}".format(validation_result))
                print("\n")
        else:
            if validation_dataset_distributed and validation_loss_fn:
                logging.info("Validation in progress at epoch end {} . . . .".format(epoch))
                with tqdm.tqdm(validation_dataset_distributed, unit=" Val batch ") as val_batches:
                    for dist_inputs in val_batches:
                        loss = strategy.run(_validate_step, args=(dist_inputs,))
                        for name, loss_value in loss.items():
                            loss_value = strategy.reduce(tf.distribute.ReduceOp.SUM, loss_value, axis=None)
                            validation_loss = validation_loss_dict_metric[name]
                            validation_loss.update_state(loss_value)

                validation_result = get_and_reset_metric_from_dict(validation_loss_dict_metric)
                write_metrics(validation_result, val_summary_writer, global_step)
                # validation_history[global_step] = validation_result
                logging.info("Validation result at epoch {} is {}".format(epoch, validation_result))
                print("\n")

    def do_callbacks(callbacks):
        """Call callbacks"""
        if not epoch_end:
            callback_scores = None
            if callbacks and callbacks_interval_steps:
                # each callback can have separate interval steps
                callback_scores = []
                for callback, callback_steps in zip(callbacks, callbacks_interval_steps):
                    if callback_steps and (global_step % callback_steps == 0):
                        logging.info("Callbacks in progress at step {} . . . .".format(global_step))
                        score = callback(trainer_kwargs)
                        callback_scores.append(score)
                    else:
                        callback_scores.append(None)
            return callback_scores
        else:
            callback_scores = None
            if callbacks:
                logging.info("Callbacks in progress at epoch end {} . . . .".format(epoch))
                callback_scores = []
                for callback in callbacks:
                    score = callback(trainer_kwargs)
                    callback_scores.append(score)

                    # Try to write a callback scores (only on epoch end)
                    # If we are returning a dict like {'exact_match': 81} or
                    # {'rougue-1': 30} etc . . . .
                    if score and isinstance(score, dict):
                        write_metrics(score, val_summary_writer, epoch)
            return callback_scores

    # Loop starts here
    # Get Tensorboard writers
    train_summary_writer, val_summary_writer = get_tensorboard_writers(model_checkpoint_dir)
    validation_history = {}
    training_history = {}
    global_step = 0
    epoch_end = False
    total_examples_processed = 0
    STEPS = steps_per_epoch // steps_per_call
    for epoch in range(1, epochs + 1):
        # start_epoch_time = time.time()
        with tqdm.trange(STEPS, unit="batch ") as tepoch:
            for step in tepoch:
                steps_covered = (step + 1) * steps_per_call
                global_step += steps_per_call
                tepoch.set_description(
                    "Epoch {}/{} --- Step {}/{} --- total examples {}".format(
                        epoch, epochs, steps_covered, steps_per_epoch, total_examples_processed
                    )
                )
                # Call Train
                do_train(train_dataset_iter)
                total_examples_processed += steps_per_call * GLOBAL_BATCH_SIZE

                # Call Validation
                do_validation(validation_dataset_distributed)

                # Call Callbacks
                callback_scores = do_callbacks(callbacks)

                # Train Metrics
                training_result = get_and_reset_metric_from_dict(training_loss_dict_metric)
                training_history[global_step] = training_result

                write_metrics(training_result, train_summary_writer, global_step)
                # training_result["learning_rate"] = learning_rate_holder.result().numpy()
                # learning_rate_holder.reset_states()
                tepoch.set_postfix(**training_result)

                # Save model
                save_model()

        # Do after every epoch
        epoch_end = True
        save_model(epoch_end)
        do_validation(validation_dataset_distributed)
        callback_scores = do_callbacks(callbacks)
        epoch_end = False

    # Flatten the results
    training_history = flat_metric_dict(training_history)
    validation_history = flat_metric_dict(validation_history)
    return training_history, validation_history, callback_scores


class TPUTrainer:
    def __init__(self, tpu_address=None, dtype='fp32'):

        distribution_strategy = 'tpu'
        allowed_dtypes = ['fp32', 'bf16']
        if dtype not in allowed_dtypes:
            raise ValueError("dtype not in {}".format(allowed_dtypes))
        self.distribution_strategy = get_distribution_strategy(
            distribution_strategy=distribution_strategy,
            tpu_address=tpu_address,
        )

        self.num_replicas = self.distribution_strategy.num_replicas_in_sync
        self._dtype = get_tf_dtype(dtype)

        # Setting dtype policy
        set_mixed_precision_policy(self._dtype)
        # # TODO
        # if self.use_tpu:
        # params["num_replicas"] = self.distribution_strategy.num_replicas_in_sync
        # else:
        # logging.info("Running transformer with num_gpus = %d", num_gpus)

        # Add keras utils threads

    def run(
        self,
        model_fn,
        optimizer_fn,
        train_dataset,
        train_loss_fn,
        epochs,
        steps_per_epoch,
        model_checkpoint_dir,
        batch_size,
        training_loss_names=None,
        validation_loss_names=None,
        validation_dataset=None,
        validation_loss_fn=None,
        validation_interval_steps=None,
        steps_per_call=100,
        enable_xla=False,
        callbacks=None,
        callbacks_interval_steps=None,
        overwrite_checkpoint_dir=False,
        max_number_of_models=10,
        model_save_interval_steps=None,
        repeat_dataset=True,
        latest_checkpoint=None,
    ):

        if steps_per_epoch:
            logging.info("Make sure `steps_per_epoch` should be less than or equal to number of batches in dataset.")
        if callbacks:
            if callbacks_interval_steps is None:
                callbacks_interval_steps = [None for callback in callbacks]
            assert len(callbacks) == len(callbacks_interval_steps)

        # Enable XLA
        keras_utils.set_session_config(enable_xla=enable_xla)
        logging.info("Policy: ----> {}".format(keras_utils.get_policy_name()))
        logging.info("Strategy: ---> {}".format(self.distribution_strategy))
        logging.info("Num TPU Devices: ---> {}".format(self.distribution_strategy.num_replicas_in_sync))

        tf.keras.backend.clear_session()
        # Under Strategy Scope
        with self.distribution_strategy.scope():
            # Model
            model = model_fn()
            # Optimizer
            optimizer = optimizer_fn()

        # We use this to avoid inferring names from loss functions
        _training_loss_names = ['loss']
        _validation_loss_names = ['loss']
        if training_loss_names:
            _training_loss_names += training_loss_names
        if validation_loss_names:
            _validation_loss_names += validation_loss_names
        # Make unique names
        training_loss_names = list(set(_training_loss_names))
        validation_loss_names = list(set(_validation_loss_names))

        # Checkpoint manager
        checkpoint_manager = save_model_checkpoints(
            model, overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models, opt=optimizer
        )

        # Try to load latest checkpoint
        model.load_checkpoint(checkpoint_dir=model_checkpoint_dir, checkpoint_path=latest_checkpoint, opt=optimizer)

        # Get metric dicts before distributing the dataset
        # ddistributed datasets has no attribute .take
        training_loss_dict_metric, validation_loss_dict_metric = get_loss_metric_dict(
            training_loss_names, validation_loss_names
        )
        # Distribute dataset
        if not repeat_dataset:
            train_dataset_distributed = self.distribution_strategy.experimental_distribute_dataset(
                train_dataset.repeat(epochs + 1)
            )
        else:
            train_dataset_distributed = self.distribution_strategy.experimental_distribute_dataset(
                train_dataset.repeat()
            )
        validation_dataset_distributed = None
        if validation_dataset:
            validation_dataset_distributed = self.distribution_strategy.experimental_distribute_dataset(
                validation_dataset
            )

        # Make train dataset iterator
        train_dataset_distributed = iter(train_dataset_distributed)

        history = {}
        training_history, validation_history, callback_scores = train_and_eval(
            model,
            optimizer,
            self.distribution_strategy,
            epochs,
            steps_per_epoch,
            steps_per_call,
            train_dataset_distributed,
            train_loss_fn,
            batch_size,
            training_loss_dict_metric,
            validation_dataset_distributed,
            validation_loss_fn,
            validation_loss_dict_metric,
            validation_interval_steps,
            callbacks,
            callbacks_interval_steps,
            locals(),
            checkpoint_manager,
            model_checkpoint_dir,
            model_save_interval_steps,
        )
        history['training_history'] = training_history
        history['validation_history'] = validation_history
        history['callbacks'] = callback_scores
        return history
