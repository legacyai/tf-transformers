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
import tensorflow as tf
import tqdm
from absl import logging

from tf_transformers.core import keras_utils
from tf_transformers.core.distribute_utils import get_distribution_strategy
from tf_transformers.core.performance_utils import (
    configure_optimizer,
    get_tf_dtype,
    is_float16,
    set_mixed_precision_policy,
)


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


def save_model_checkpoints(model, overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models):
    # Model checkpoint
    if not overwrite_checkpoint_dir:
        import os

        if os.path.exists(model_checkpoint_dir):
            raise FileExistsError("Model directory exists")

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_checkpoint_dir, max_to_keep=max_number_of_models)
    return manager


def get_loss_metric_dict(model, dataset, loss_fn, validation_dataset, validation_loss_fn):
    for (batch_inputs, batch_labels) in dataset.take(1):
        model_outputs = model(batch_inputs)
        train_loss_dict = loss_fn(batch_labels, model_outputs)
        training_loss_dict_metric = {name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in train_loss_dict}

    training_loss_dict_metric["learning_rate"] = tf.keras.metrics.Mean(
        "learning_rate", dtype=tf.float32
    )  # We store learning rate here and reset after every global steps

    validation_loss_dict_metric = {}
    if validation_dataset and validation_loss_fn:
        for (batch_inputs, batch_labels) in dataset.take(1):
            model_outputs = model(batch_inputs)
            valid_loss_dict = validation_loss_fn(batch_labels, model_outputs)
            validation_loss_dict_metric = {
                name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in valid_loss_dict
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
    train_log_dir = model_checkpoint_dir + "/logs/train"
    test_log_dir = model_checkpoint_dir + "/logs/dev"
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
    mixed_precision,
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

    def write_metrics(metric_dict, writer, step):
        @tf.function
        def _write(step):
            # other model code would go here
            with writer.as_default():
                for name, result in metric_dict.items():
                    tf.summary.scalar(name, result, step=step)

        _write(step)
        writer.flush()

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
                if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                    loss_scaled = {name: optimizer.get_scaled_loss(loss_value) for name, loss_value in loss.items()}
                # TODO
                # Scales down the loss for gradients to be invariant from replicas.
                # loss = loss / strategy.num_replicas_in_sync
            if mixed_precision:
                scaled_gradients = tape.gradient(loss_scaled["loss"], model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                grads = tape.gradient(loss["loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # training_loss.update_state(loss * strategy.num_replicas_in_sync)
            return loss

        for _ in tf.range(tf.convert_to_tensor(steps_per_call)):
            dist_inputs = next(iterator)
            loss = strategy.run(train_step, args=(dist_inputs,))
            # strategy reduce
            loss = {
                name: strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_value, axis=None)
                for name, loss_value in loss.items()
            }
            for name, loss_value in loss.items():
                training_loss = training_loss_dict_metric[name]
                training_loss.update_state(loss_value)
            # Get current learning rate
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                current_lr = optimizer._optimizer._decayed_lr(tf.float32)
            else:
                current_lr = optimizer._decayed_lr(tf.float32)
            training_loss_dict_metric["learning_rate"].update_state(current_lr)
            # training_result = get_and_reset_metric_from_dict(training_loss_dict_metric)

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
                            loss_value = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_value, axis=None)
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
                            loss_value = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_value, axis=None)
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
                logging.info("Callbacks in progress at step {} . . . .".format(global_step))
                callback_scores = []
                for callback, callback_steps in zip(callbacks, callbacks_interval_steps):
                    if callback_steps and (global_step % callback_steps == 0):
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
            return callback_scores

    # Loop starts here
    # Get Tensorboard writers
    train_summary_writer, val_summary_writer = get_tensorboard_writers(model_checkpoint_dir)
    validation_history = {}
    training_history = {}
    global_step = 0
    epoch_end = False
    STEPS = steps_per_epoch // steps_per_call
    for epoch in range(1, epochs + 1):
        # start_epoch_time = time.time()
        with tqdm.trange(STEPS, unit="batch ") as tepoch:
            for step in tepoch:
                steps_covered = (step + 1) * steps_per_call
                global_step += steps_per_call
                tepoch.set_description(
                    "Epoch {}/{} --- Step {}/{} --- ".format(epoch, epochs, steps_covered, steps_per_epoch)
                )
                # Call Train
                do_train(train_dataset_iter)

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


class Trainer:
    def __init__(
        self,
        distribution_strategy,
        num_gpus=0,
        all_reduce_alg=None,
        num_packs=1,
        tpu_address=None,
        dtype='fp32',
        loss_scale='dynamic',
    ):

        self.distribution_strategy = get_distribution_strategy(
            distribution_strategy=distribution_strategy,
            num_gpus=num_gpus,
            all_reduce_alg=all_reduce_alg,
            num_packs=num_packs,
            tpu_address=tpu_address,
        )

        self.num_replicas = self.distribution_strategy.num_replicas_in_sync
        self._dtype = get_tf_dtype(dtype)

        # Setting dtype policy
        set_mixed_precision_policy(self._dtype)
        self.use_float16 = is_float16(self._dtype)
        self.loss_scale = loss_scale

        # # TODO
        # if self.use_tpu:
        # params["num_replicas"] = self.distribution_strategy.num_replicas_in_sync
        # else:
        # logging.info("Running transformer with num_gpus = %d", num_gpus)

        # Add keras utils threads

    @property
    def use_tpu(self):
        if self.distribution_strategy:
            return isinstance(self.distribution_strategy, tf.distribute.TPUStrategy)
        return False

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
        validation_dataset=None,
        validation_loss_fn=None,
        validation_interval_steps=None,
        steps_per_call=100,
        enable_xla=True,
        callbacks=None,
        callbacks_interval_steps=None,
        overwrite_checkpoint_dir=False,
        max_number_of_models=10,
        model_save_interval_steps=None,
        repeat_dataset=False,
    ):

        if steps_per_epoch:
            logging.info("Make sure `steps_per_epoch` should be less than or equal to number of batches in dataset.")
        if callbacks:
            assert len(callbacks) == len(callbacks_interval_steps)

        # Enable XLA
        keras_utils.set_session_config(enable_xla=enable_xla)
        logging.info("Policy: ----> {}".format(keras_utils.get_policy_name()))
        logging.info("Strategy: ---> {}".format(self.distribution_strategy))
        if self.use_tpu:
            logging.info("Num TPU Devices: ---> {}".format(self.distribution_strategy.num_replicas_in_sync))
        else:
            logging.info("Num GPU Devices: ---> {}".format(self.distribution_strategy.num_replicas_in_sync))

        # Under Strategy Scope
        with self.distribution_strategy.scope():
            # Model
            model = model_fn()

            # Optimizer
            optimizer = optimizer_fn()
            # TPU do not need this
            if not self.use_tpu:
                optimizer = configure_optimizer(optimizer, use_float16=self.use_float16, loss_scale=self.loss_scale)

        # Checkpoint manager
        checkpoint_manager = save_model_checkpoints(
            model, overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models
        )

        # Get metric dicts before distributing the dataset
        # ddistributed datasets has no attribute .take
        logging.info("Inferring metric shapes . . . . .")
        training_loss_dict_metric, validation_loss_dict_metric = get_loss_metric_dict(
            model, train_dataset, train_loss_fn, validation_dataset, validation_loss_fn
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
            self.use_float16,
            callbacks,
            callbacks_interval_steps,
            locals(),
            checkpoint_manager,
            model_checkpoint_dir,
            model_save_interval_steps,
        )
        history['training_history'] = training_history
        history['validation_hsitory'] = validation_history
        history['callbacks'] = callback_scores
        return history
