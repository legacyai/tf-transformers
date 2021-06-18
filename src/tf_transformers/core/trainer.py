import time
from pprint import pformat

import tensorflow as tf
import tqdm
from absl import logging

logging.set_verbosity("INFO")


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
        pass
    model_outputs = model(batch_inputs)
    train_loss_dict = loss_fn(batch_labels, model_outputs)
    training_loss_dict_metric = {name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in train_loss_dict}

    validation_loss_dict_metric = {}
    if validation_dataset and validation_loss_fn:
        for (batch_inputs, batch_labels) in dataset.take(1):
            pass
        model_outputs = model(batch_inputs)
        valid_loss_dict = validation_loss_fn(batch_labels, model_outputs)
        validation_loss_dict_metric = {name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in valid_loss_dict}

    return training_loss_dict_metric, validation_loss_dict_metric


def get_and_reset_metric_from_dict(metric_dict):
    if not metric_dict:
        return {}
    metric_result = {name: metric.result().numpy() for name, metric in metric_dict.items()}
    for name, metric in metric_dict.items():
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


def SingleDeviceTrainer(
    model,
    dataset,
    loss_fn,
    optimizer,
    epochs,
    validation_dataset,
    validation_loss_fn,
    model_checkpoint_dir,
    steps_per_epoch,
    mixed_precision=False,
    max_number_of_models=10,
    model_save_interval_steps=1000,
    overwrite_checkpoint_dir=False,
    validation_interval_steps=None,
    steps_per_call=100,
    eval_callbacks=None,
    skip_pre_train_validation=True,
):
    """SingleDevice Trainer

    Args:
        model (LegacyModel/tf.keras.Model): Model object.
        dataset (tf.data.Dataset): Tensorlow Dataset (not iterator)
        loss_fn (loss_fn): a function which must return only dict, with 'loss' key reserved for loss to minimize.
        optimizer ([tf.keras.optimizers]): Optimizer
        epochs ([int]): Total number of epochs
        validation_dataset ([optional]): Tensorlow Dataset (not iterator)
        validation_loss_fn ([validation_loss_fn]): a function which must return only dict
        model_checkpoint_dir ([str]): Directory to save model
        steps_per_epoch ([int]): Number of batch_data to loop over per epoch.
        mixed_precision (bool, optional): [Mixed precision]. Defaults to False.
        max_number_of_models (int, optional): [Total number of models to keep in a directory]. Defaults to 10.
        model_save_interval_steps (int, optional): [Steps]. Defaults to 1000.
        overwrite_checkpoint_dir (bool, optional): [Whether to overwrite existing directory or not]. Defaults to False.
        validation_interval_steps ([type], optional): [Steps]. Defaults to None.
        steps_per_call (int, optional): [Number of steps inside @tf.function per step]. Defaults to 100.
        eval_callbacks ([type], optional): [List of call backs]. Defaults to None.

        eval_callbacks should be designed in such a way that it accepts only kwargs from this trainer
        inside call: Look at examples.

    Returns:
        [dict]: History
    """

    if steps_per_epoch:
        logging.info("Make sure `steps_per_epoch` should be less than or equal to number of batches in dataset.")
    checkpoint_manager = save_model_checkpoints(
        model, overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models
    )

    if mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Train Functions
    @tf.function
    def train(iterator):
        """The step function for one training step"""

        def train_step(batch_inputs, batch_labels):
            """The computation to run on each TPU device."""
            with tf.GradientTape() as tape:
                model_outputs = model(batch_inputs)
                loss = loss_fn(batch_labels, model_outputs)

                if mixed_precision:
                    loss = {name: optimizer.get_scaled_loss(loss_value) for name, loss_value in loss.items()}
            if mixed_precision:
                scaled_gradients = tape.gradient(loss["loss"], model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                grads = tape.gradient(loss["loss"], model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            # training_loss.update_state(loss * strategy.num_replicas_in_sync)

            for name, loss_value in loss.items():
                training_loss = training_loss_dict_metric[name]
                training_loss.update_state(loss_value)
            current_lr = optimizer._decayed_lr(tf.float32)
            learning_rate_holder.update_state(current_lr)

        for _ in tf.range(tf.convert_to_tensor(steps_per_call)):
            batch_inputs, batch_labels = next(iterator)
            train_step(batch_inputs, batch_labels)

    # Validate Functions
    @tf.function
    def _validate(validation_dataset):
        """Validation step"""
        for (batch_inputs, batch_labels) in validation_dataset:
            model_outputs = model(batch_inputs)
            loss = validation_loss_fn(batch_labels, model_outputs)
            for name, loss_value in loss.items():
                validation_loss = validation_loss_dict_metric[name]
                validation_loss.update_state(loss_value)

    # do validation
    def do_validation(validation_dataset, trainer_kwargs):
        if validation_dataset and validation_loss_fn:
            _validate(validation_dataset)
        validation_result = get_and_reset_metric_from_dict(validation_loss_dict_metric)
        callback_scores = []
        if eval_callbacks:
            for i, eval_callback in enumerate(eval_callbacks):
                score = eval_callback(trainer_kwargs)
                callback_scores.append(score)
            validation_result["callback_score"] = callback_scores
        return validation_result

    # Metrics
    learning_rate_holder = tf.keras.metrics.Mean(
        "learning_rate_holder", dtype=tf.float32
    )  # We store learning rate here and reset after every global steps
    training_loss_dict_metric, validation_loss_dict_metric = get_loss_metric_dict(
        model, dataset, loss_fn, validation_dataset, validation_loss_fn
    )
    # dataset to iterator
    dataset_iterator = iter(dataset.repeat(epochs + 1))
    # Default ---> Do validation before model got trained
    if not skip_pre_train_validation:
        val_result = do_validation(validation_dataset, locals())
        print(pformat("Validation result before training {}".format(val_result)))
        logging.info(pformat("Validation result before training {}".format(val_result)))

    # Get Tensorboard writers
    train_summary_writer, val_summary_writer = get_tensorboard_writers(model_checkpoint_dir)

    # Main Loop
    STEPS = steps_per_epoch // steps_per_call
    history = {}
    train_history = {}
    validation_history = {}
    global_step = 0
    for epoch in range(1, epochs + 1):
        start_epoch_time = time.time()
        epoch_loss = []
        with tqdm.trange(STEPS, unit="batch ") as tepoch:
            for step in tepoch:
                steps_covered = (step + 1) * steps_per_call
                global_step += steps_per_call
                tepoch.set_description(
                    "Epoch {}/{} --- Step {}/{} --- ".format(epoch, epochs, steps_covered, steps_per_epoch)
                )

                # Call Train
                train(dataset_iterator)
                # Get Train metrics
                training_result = get_and_reset_metric_from_dict(training_loss_dict_metric)
                epoch_loss.append(training_result["loss"])  # To get average at the end of epoch
                training_result["learning_rate"] = learning_rate_holder.result().numpy()
                learning_rate_holder.reset_states()
                tepoch.set_postfix(**training_result)

                train_history[global_step] = training_result
                write_metrics(training_result, train_summary_writer, global_step)

                # Do after provided steps
                if validation_interval_steps:
                    if steps_covered % validation_interval_steps == 0:
                        val_result = do_validation(validation_dataset, locals())
                        validation_history[global_step] = val_result
                        write_metrics(val_result, val_summary_writer, global_step)
                        print("-----------------------------------------------------------------------")
                        logging.info(
                            pformat(
                                ("Epoch {} , Step {} , validation result {}".format(epoch, steps_covered, val_result))
                            )
                        )

                # Save model
                if model_save_interval_steps:
                    if steps_covered % model_save_interval_steps == 0:
                        checkpoint_manager.save()

        # After an epoch
        checkpoint_manager.save()
        end_epoch_time = time.time()
        val_result = do_validation(validation_dataset, locals())
        print("validation result", val_result)
        if val_result:
            print("-----------------------------------------------------------------------")
            logging.info(pformat(("Epoch {} , validation result {}".format(epoch, val_result))))
            validation_history[global_step] = val_result
            print(pformat(("Epoch {} , validation result {}".format(epoch, val_result))))
        print("-----------------------------------------------------------------------")
        logging.info(
            pformat(
                "Epoch {}/{} --- Mean Loss {} --- Time {} seconds".format(
                    epoch + 1, epochs, tf.reduce_mean(epoch_loss), end_epoch_time - start_epoch_time
                )
            )
        )
        print(
            pformat(
                "Epoch {}/{} --- Mean Loss {} --- Time {} seconds".format(
                    epoch, epochs, tf.reduce_mean(epoch_loss), end_epoch_time - start_epoch_time
                )
            )
        )

    history["train_history"] = train_history
    history["validation_history"] = validation_history

    return history
