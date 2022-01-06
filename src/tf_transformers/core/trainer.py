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
import time
from typing import Callable, List

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
from tf_transformers.utils import tf_utils


# COLORS
class Color:
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    RESET = 39


def flat_callback_scores(callback_scores: List):
    callback_scores_flatten = []
    for item in callback_scores:
        if all(v is None for v in item):
            continue
        callback_scores_flatten.append(item)

    return callback_scores_flatten


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


def get_loss_metric_dict(training_loss_names: List, validation_loss_names: List):
    """Get metric based on names"""
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


def save_model_checkpoints(model, overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models, **kwargs):
    """Return checkpoint manager"""
    if not overwrite_checkpoint_dir:
        import os

        if os.path.exists(model_checkpoint_dir):
            raise FileExistsError("Model directory exists")

    checkpoint = tf.train.Checkpoint(model=model, **kwargs)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_checkpoint_dir, max_to_keep=max_number_of_models)
    return manager


def get_and_reset_metric_from_dict(metric_dict):
    """Convert metric to dict of results and reset"""
    if not metric_dict:
        return {}
    metric_result = {name: metric.result().numpy() for name, metric in metric_dict.items()}
    for _name, metric in metric_dict.items():
        metric.reset_states()
    return metric_result


def get_tensorboard_writers(model_checkpoint_dir):
    """Tensorboard Writer"""
    train_log_dir = os.path.join(model_checkpoint_dir, "logs/train")
    test_log_dir = os.path.join(model_checkpoint_dir, "logs/eval")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer


def train_and_eval(
    model,
    optimizer,
    strategy,
    epochs,
    global_step,
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
    model_checkpoint_dir,
    model_save_interval_steps,
    max_number_of_models,
    clip_norm,
    wandb,
    ckpt_number,
):
    def save_model(checkpoint_manager, epoch_end=False):
        """Save model"""
        if checkpoint_manager is None:
            checkpoint_manager = save_model_checkpoints(
                model,
                True,
                model_checkpoint_dir,
                max_number_of_models,
                opt=optimizer,
                step=tf.Variable(global_step, dtype=tf.int64),
            )
        if not epoch_end:
            if model_save_interval_steps:
                if global_step % model_save_interval_steps == 0:
                    checkpoint_manager.save(checkpoint_number=epoch)
                    logging.info(
                        "Model saved at step {} at {}".format(global_step, checkpoint_manager.latest_checkpoint)
                    )
        else:
            checkpoint_manager.save(checkpoint_number=epoch)
            logging.info("Model saved at epoch {} at {}".format(epoch, checkpoint_manager.latest_checkpoint))

    def write_metrics(metric_dict, writer, step):
        """Write metrics here"""

        def _write(step):
            # other model code would go here
            with writer.as_default():
                for name, result in metric_dict.items():
                    tf.summary.scalar(name, result, step=step)

        _write(step)
        writer.flush()

    def write_metrics_to_wandb(metric_dict, wandb_writer, step):
        """Write metrics here"""
        # Write if wandb is not None
        if wandb:
            wandb_writer.log(metric_dict, step=step)

    def write_validation_metrics_to_wandb(metric_dict, wandb_writer, step):
        """Write metrics here"""
        # Write if wandb is not None
        metric_dict_copy = {}
        for k, v in metric_dict.items():
            if k.startswith("val"):
                metric_dict_copy[k] = v
            else:
                metric_dict_copy['val_' + k] = v

        if wandb:
            wandb_writer.log(metric_dict_copy, step=step)

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

            # per_example_loss_averaged[name] = tf.reduce_sum(per_example_loss_averaged[name])
        return per_example_loss_averaged

    # Callbacks
    def do_callbacks(callbacks):
        """Call callbacks"""
        if not epoch_end:
            callback_scores = None
            if callbacks and callbacks_interval_steps:
                # each callback can have separate interval steps
                callback_scores = []
                for callback, callback_steps in zip(callbacks, callbacks_interval_steps):
                    if callback_steps and global_step != 0 and (global_step % callback_steps == 0):
                        logging.info("Callbacks in progress at step {} . . . .".format(global_step))
                        current_trainer_kwargs = locals()
                        trainer_kwargs.update(current_trainer_kwargs)
                        score = callback(trainer_kwargs)
                        logging.info("Callback score {} at step {}".format(score, global_step))
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
                    current_trainer_kwargs = locals()
                    trainer_kwargs.update(current_trainer_kwargs)
                    score = callback(trainer_kwargs)
                    callback_scores.append(score)
                    logging.info("Callback score {} at epoch {}".format(score, epoch))
                    # Try to write a callback scores (only on epoch end)
                    # If we are returning a dict like {'exact_match': 81} or
                    # {'rogue-1': 30} etc . . . .
                    if score and isinstance(score, dict):
                        # Write to tensorboard
                        write_metrics(score, val_summary_writer, epoch)
                        # Write to Wandb
                        write_metrics_to_wandb(score, wandb, epoch)
            return callback_scores

    # Validation Functions
    # Infer number of steps for validation data, otherwise
    # it will run indefinitely inside tf.function
    validation_steps = 0
    if validation_dataset_distributed:
        for _ in validation_dataset_distributed:
            validation_steps += 1

    @tf.function
    def _do_validation(dataset):
        """The step function for one validation step"""

        def validate_step(dist_inputs):
            """The computation to run on each device."""
            batch_inputs, batch_labels = dist_inputs
            model_outputs = model(batch_inputs)
            loss = compute_loss_valid(batch_labels, model_outputs)
            return loss

        for dist_inputs in dataset:
            loss = strategy.run(validate_step, args=(dist_inputs,))
            # strategy reduce (SUM) is important
            # If not SUM, final loss might not be a good representative of global batch
            loss = {
                name: strategy.reduce(tf.distribute.ReduceOp.SUM, loss_value, axis=None)
                for name, loss_value in loss.items()
            }
            for name, loss_value in loss.items():
                # get loss metric object based on loss names ('loss', 'loss1' etc . . . .)
                validation_loss_metric = validation_loss_dict_metric[name]
                validation_loss_metric.update_state(loss_value)

    def do_validation(validation_dataset_distributed):
        """Batch validation"""
        with tqdm.trange(validation_steps, unit=" Validation batch ", colour='blue') as val_bar:
            step_counter = 0
            _do_validation(validation_dataset_distributed)
            val_bar.set_description(
                "Epoch {}/{} --- Val Step {}/{} ".format(epoch, epochs, step_counter, validation_steps)
            )

            validation_result = get_and_reset_metric_from_dict(validation_loss_dict_metric)
            val_bar.set_postfix(**validation_result)
        return validation_result

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
                # Means we are in GPU fp16 mixed precision
                if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                    loss_scaled = {name: optimizer.get_scaled_loss(loss_value) for name, loss_value in loss.items()}
                    scaled_gradients = tape.gradient(loss_scaled["loss"], model.trainable_variables)
                    grads = optimizer.get_unscaled_gradients(scaled_gradients)
                    if clip_norm:
                        # Apply some clipping
                        grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                else:
                    grads = tape.gradient(loss["loss"], model.trainable_variables)
                    if clip_norm:
                        # Apply some clipping
                        grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                # TODO
                # Scales down the loss for gradients to be invariant from replicas.
                # loss = loss / strategy.num_replicas_in_sync
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
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                current_lr = optimizer._optimizer._decayed_lr(tf.float32)
            else:
                current_lr = optimizer._decayed_lr(tf.float32)
            training_loss_dict_metric["learning_rate"].update_state(current_lr)

    # Loop starts here
    # Get Tensorboard writers
    train_summary_writer, val_summary_writer = get_tensorboard_writers(model_checkpoint_dir)
    validation_history = {}
    training_history = {}
    all_callback_scores = []
    epoch_end = False
    total_examples_processed = 0
    checkpoint_manager = None  # Define it to be None here . Check save_model.
    STEPS = steps_per_epoch // steps_per_call
    epochs = epochs + ckpt_number
    for epoch in range(ckpt_number, epochs):
        # start_epoch_time = time.time()
        with tqdm.trange(STEPS, unit="batch ", colour='green') as tepoch:
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
                if not epoch_end:
                    if (
                        validation_dataset_distributed
                        and validation_loss_fn
                        and validation_interval_steps
                        and (global_step % validation_interval_steps == 0)
                    ):
                        # Do validation and get result
                        validation_result = do_validation(validation_dataset_distributed)
                        # Add to history
                        validation_history[global_step] = validation_result
                        # Write to tensorboard
                        write_metrics(validation_result, val_summary_writer, global_step)
                        # Write to Wandb
                        write_validation_metrics_to_wandb(validation_result, wandb, global_step)

                # Call Callbacks
                callback_scores = do_callbacks(callbacks)
                if callback_scores:
                    all_callback_scores.append(callback_scores)

                # Train Metrics
                # Do training and get result
                training_result = get_and_reset_metric_from_dict(training_loss_dict_metric)
                # Add to history
                training_history[global_step] = training_result
                # Write to tensorboard
                write_metrics(training_result, train_summary_writer, global_step)
                # Write to Wandb
                write_metrics_to_wandb(training_result, wandb, global_step)
                # training_result["learning_rate"] = learning_rate_holder.result().numpy()
                # learning_rate_holder.reset_states()
                tepoch.set_postfix(**training_result)

                # Save model
                save_model(checkpoint_manager)

        # Do after every epoch
        epoch_end = True
        save_model(checkpoint_manager, epoch_end)
        print()
        time.sleep(0.1)  # Sleep for 1 second, to make print neater

        validation_result = None
        if validation_dataset_distributed and validation_loss_fn:
            # Do validation and get result
            validation_result = do_validation(validation_dataset_distributed)
            # Add to history
            validation_history[global_step] = validation_result
            # Write to tensorboard
            write_metrics(validation_result, val_summary_writer, global_step)
            # Write to Wandb
            write_validation_metrics_to_wandb(validation_result, wandb, global_step)
            logging.info(
                "Validation result at epcoh {} and \
                global step {} is {}".format(
                    epoch, global_step, validation_result
                )
            )
        print()
        time.sleep(0.1)  # Sleep for 1 second, to make print neater
        callback_scores = do_callbacks(callbacks)
        if callback_scores:
            all_callback_scores.append(callback_scores)
        epoch_end = False
        print()
        time.sleep(0.1)  # Sleep for 1 second, to make print neater

    # Flatten the results
    training_history = flat_metric_dict(training_history)
    validation_history = flat_metric_dict(validation_history)
    all_callback_scores = flat_callback_scores(all_callback_scores)
    return training_history, validation_history, all_callback_scores


class Trainer:
    """Trainer for the Models"""

    def __init__(
        self,
        distribution_strategy: str,
        tpu_address: str = None,
        dtype: str = 'fp32',
        num_gpus: int = 0,
        all_reduce_alg: str = None,
        num_packs: int = 1,
        loss_scale: str = 'dynamic',
    ):
        """Trainer class

        Args:
            distribution_strategy (:obj:`str`): a string specifying which distribution strategy to
            use. Accepted values are :obj:`("off", "one_device", "mirrored",
            "parameter_server", "multi_worker_mirrored", "cpu", and "tpu")` -- case
            insensitive. "tpu" means to use TPUStrategy using `tpu_address`.
            "off" means to use the default strategy which is obtained from
            tf.distribute.get_strategy (for details on the default strategy, see
            https://www.tensorflow.org/guide/distributed_training#default_strategy)

            tpu_address (:obj:`str`, `optional`, defaults to None): If you training in cloud TPU VM,
            you can pass :obj:(`local`). If you are connecting to TPU from a different machine, provide
            the name of the machine.
            dtype (:obj:`str`, `optional`, defaults to fp32): The dtype for training. Supported
            values are :obj:`(fp32, fp16)` for GPU and :obj:`(fp32, bf16)` for TPU.
            num_gpus (:obj:`int`, `optional`, defaults to 0): Number of GPUs.
            all_reduce_alg (:obj:`str`, `optional`, defaults to None): Supported values are
            :obj:`(ring, nccl)`.
            num_packs (:obj:`int`, `optional`, defaults to 1): an integer specifying number of packs
            for the cross device op.
            loss_scale (:obj:`str`, `optional`, defaults to dynamic): Loss scaling for optimizer in the case
            of dtype=fp16.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        self.distribution_strategy_name = distribution_strategy
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

    def run(
        self,
        model_fn: Callable,
        optimizer_fn: Callable,
        train_dataset: tf.data.Dataset,
        train_loss_fn: Callable,
        epochs: int,
        steps_per_epoch: int,
        model_checkpoint_dir: str,
        batch_size: int,
        training_loss_names: List = None,
        validation_loss_names: List = None,
        validation_dataset: tf.data.Dataset = None,
        validation_loss_fn: Callable = None,
        validation_interval_steps: int = None,
        steps_per_call: int = 100,
        enable_xla: bool = False,
        callbacks: List = None,
        callbacks_interval_steps: List = None,
        max_number_of_models: int = 100,
        model_save_interval_steps: bool = None,
        repeat_dataset: bool = True,
        latest_checkpoint: str = None,
        clip_norm=None,
        wandb=None,
    ):

        if steps_per_epoch:
            logging.info("Make sure `steps_per_epoch` should be less than or equal to number of batches in dataset.")
        if callbacks:
            # We want `callbacks` and `callbacks_interval_steps` to be list.
            if callbacks_interval_steps is None:
                callbacks_interval_steps = [None for callback in callbacks]
            assert len(callbacks) == len(callbacks_interval_steps)

        # Enable XLA
        if enable_xla:
            # Enable XLA
            keras_utils.set_session_config(enable_xla=enable_xla)

        # This should be outisde the distribution scope (important)
        # TODO: From 2.5 onwards this has to be disabled
        # tf.keras.backend.clear_session()

        # Log info
        logging.info("Policy: ----> {}".format(keras_utils.get_policy_name()))
        logging.info("Strategy: ---> {}".format(self.distribution_strategy))
        _is_gpu_available, _num_gpus_present = tf_utils.is_gpu_available()
        if self.distribution_strategy_name == 'tpu':
            logging.info("Num TPU Devices: ---> {}".format(self.distribution_strategy.num_replicas_in_sync))
        else:
            if _is_gpu_available:
                logging.info("Num GPU Devices: ---> {}".format(self.distribution_strategy.num_replicas_in_sync))
            else:
                logging.info("Num CPU Devices: ---> {}".format(self.distribution_strategy.num_replicas_in_sync))

        # We use this to avoid inferring names from loss functions
        # We need this names for metrics and tensorboards.
        _training_loss_names = ['loss']
        _validation_loss_names = ['loss']
        if training_loss_names:
            _training_loss_names += training_loss_names
        if validation_loss_names:
            _validation_loss_names += validation_loss_names
        # Make unique names
        training_loss_names = list(set(_training_loss_names))
        validation_loss_names = list(set(_validation_loss_names))

        # Under Strategy Scope
        with self.distribution_strategy.scope():
            # Model
            model = model_fn()

            # Optimizer
            optimizer = optimizer_fn()

            # Only for GPU fp16
            if self.use_float16:
                # Configure Optimizer for fp16 if True
                optimizer = configure_optimizer(optimizer, use_float16=self.use_float16, loss_scale=self.loss_scale)

            # Load and restore
            ckpt = model.load_checkpoint(
                checkpoint_dir=model_checkpoint_dir,
                checkpoint_path=latest_checkpoint,
                opt=optimizer,
                step=tf.Variable(0, dtype=tf.int64),
            )

        # If checkpoint is not None, means its succesful
        global_step = 0
        ckpt_number = 1
        if ckpt:
            global_step = ckpt.step.numpy()
            optimizer = ckpt.opt
            logging.info("Succesfully restored existing checkpoints from step {}".format(global_step))
            # Extract number from latest checkpoint
            if latest_checkpoint:
                ckpt_number = int(latest_checkpoint.split("/")[-1].replace("ckpt-", "").strip())
                ckpt_number += 1
            else:
                latest_checkpoint = tf.train.latest_checkpoint(model_checkpoint_dir)
                ckpt_number = int(latest_checkpoint.split("/")[-1].replace("ckpt-", "").strip())
                ckpt_number += 1

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

        # Get metric dicts before distributing the dataset
        # ddistributed datasets has no attribute .take
        training_loss_dict_metric, validation_loss_dict_metric = get_loss_metric_dict(
            training_loss_names, validation_loss_names
        )

        history = {}
        training_history, validation_history, callback_scores = train_and_eval(
            model,
            optimizer,
            self.distribution_strategy,
            epochs,
            global_step,
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
            model_checkpoint_dir,
            model_save_interval_steps,
            max_number_of_models,
            clip_norm,
            wandb,
            ckpt_number,
        )
        history['training_history'] = training_history
        history['validation_history'] = validation_history
        history['callbacks'] = callback_scores

        # Save json
        return history
