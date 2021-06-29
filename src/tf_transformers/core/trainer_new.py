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


def write_metrics(metric_dict, writer, step):
    with writer.as_default():
        for name, result in metric_dict.items():
            tf.summary.scalar(name, result, step=step)


def train_and_eval(
    model,
    optimizer,
    strategy,
    epochs,
    steps_per_epoch,
    steps_per_call,
    train_dataset_iter,
    train_loss_fn,
    training_loss_dict_metric,
    validation_dataset_distributed,
    validation_loss_fn,
    validation_loss_dict_metric,
    validation_interval_steps,
    mixed_precision,
    callbacks,
    trainer_kwargs,
):

    # Train Functions
    @tf.function
    def do_train(iterator):
        """The step function for one training step"""

        def train_step(dist_inputs):
            """The computation to run on each device."""
            batch_inputs, batch_labels = dist_inputs
            with tf.GradientTape() as tape:
                model_outputs = model(batch_inputs)
                loss = train_loss_fn(batch_labels, model_outputs)
                # TODO
                # Scales down the loss for gradients to be invariant from replicas.
                # loss = loss / strategy.num_replicas_in_sync
            if mixed_precision:
                loss = {name: optimizer.get_scaled_loss(loss_value) for name, loss_value in loss.items()}
                scaled_gradients = tape.gradient(loss["loss"], model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                grads = tape.gradient(loss["loss"], model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            # training_loss.update_state(loss * strategy.num_replicas_in_sync)

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
            current_lr = optimizer._decayed_lr(tf.float32)
            training_loss_dict_metric["learning_rate"].update_state(current_lr)

    # do validation
    @tf.function()
    def do_validation(validation_dataset_distributed):
        """Validation step"""

        def _validate_step(dist_inputs):

            batch_inputs, batch_labels = dist_inputs
            model_outputs = model(batch_inputs)
            loss = validation_loss_fn(batch_labels, model_outputs)
            for name, loss_value in loss.items():
                loss_value = strategy.reduce(tf.distribute.ReduceOp.SUM, loss_value, axis=None)
                validation_loss = validation_loss_dict_metric[name]
                validation_loss.update_state(loss_value)

        if validation_dataset_distributed and (global_step + 1 % validation_interval_steps == 0):
            for dist_inputs in validation_dataset_distributed:
                strategy.run(_validate_step, args=(dist_inputs,))

            validation_result = get_and_reset_metric_from_dict(validation_loss_dict_metric)
            validation_history[global_step] = validation_result

    def do_callbacks(callbacks):
        """Call callbacks"""
        callback_scores = None
        if callbacks and (global_step + 1 % validation_interval_steps == 0):
            callback_scores = []
            for callback in callbacks:
                score = callback(trainer_kwargs)
                callback_scores.append(score)
        return callback_scores

    # Loop starts here
    validation_history = {}
    training_history = {}
    global_step = 0
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
                do_callbacks(callbacks)

                # Train Metrics
                training_result = get_and_reset_metric_from_dict(training_loss_dict_metric)
                # training_result["learning_rate"] = learning_rate_holder.result().numpy()
                # learning_rate_holder.reset_states()
                tepoch.set_postfix(**training_result)

    return training_history, validation_history


class TrainerNew:
    def __init__(
        self,
        distribution_strategy,
        num_gpus,
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
        validation_dataset=None,
        validation_loss_fn=None,
        validation_interval_steps=None,
        steps_per_call=100,
        enable_xla=True,
        callbacks=None,
    ):

        logging.info("Run")
        # Enable XLA
        keras_utils.set_session_config(enable_xla=enable_xla)

        # Under Strategy Scope
        with self.distribution_strategy.scope():
            # Model
            model = model_fn()

            # Optimizer
            optimizer = optimizer_fn()
            optimizer = configure_optimizer(optimizer, use_float16=self.use_float16, loss_scale=self.loss_scale)

        # Get metric dicts before distributing the dataset
        # ddistributed datasets has no attribute .take
        training_loss_dict_metric, validation_loss_dict_metric = get_loss_metric_dict(
            model, train_dataset, train_loss_fn, validation_dataset, validation_loss_fn
        )
        # Distribute dataset
        train_dataset_distributed = self.distribution_strategy.experimental_distribute_dataset(
            train_dataset.repeat(epochs + 1)
        )
        if validation_dataset:
            validation_dataset_distributed = self.distribution_strategy.experimental_distribute_dataset(
                validation_dataset
            )

        # Make train dataset iterator
        train_dataset_distributed = iter(train_dataset_distributed)

        history = {}
        training_history, validation_history = train_and_eval(
            model,
            optimizer,
            self.distribution_strategy,
            epochs,
            steps_per_epoch,
            steps_per_call,
            train_dataset_distributed,
            train_loss_fn,
            training_loss_dict_metric,
            validation_dataset_distributed,
            validation_loss_fn,
            validation_loss_dict_metric,
            validation_interval_steps,
            self.use_float16,
            callbacks,
            locals(),
        )

        return training_history, validation_history
