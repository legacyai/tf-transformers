import tensorflow as tf
import time
import tqdm
from pprint import pformat
from absl import logging

logging.set_verbosity("INFO")


def save_model_checkpoints(overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models):
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
    metric_result = {name: metric.result().numpy()[0] for name, metric in metric_dict.items()}
    for name, metric in metric_dict.items():
        metric.reset_states()
    return metric_result


def trainer(
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
):

    if steps_per_epoch:
        logging.info("Make sure `steps_per_epoch` should be less than or equal to number of batches in dataset.")
    checkpoint_manager = save_model_checkpoints(overwrite_checkpoint_dir, model_checkpoint_dir, max_number_of_models)

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
    def do_validation(validation_dataset):
        if validation_dataset and validation_loss_fn:
            _validate(validation_dataset)
        validation_result = get_and_reset_metric_from_dict(validation_loss_dict_metric)
        if eval_callback:
            for i, eval_callback in enumerate(eval_callbacks):
                score = eval_callback(kwargs)
                validation_result["callback_score_{}".format(i)] = score
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
    if validation_dataset and validation_loss_fn:
        val_result = do_validation(validation_dataset)
        logging.info(pformat("Validation result before training {}".format(val_result)))

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
                global_step += 1
                steps_covered = (step + 1) * steps_per_call
                tepoch.set_description(
                    "Epoch {}/{} --- Step {}/{} --- ".format(epoch, epochs, steps_covered, steps_per_epoch)
                )

                # Call Train
                train(dataset_iterator)
                # Get Train metrics
                training_result = get_and_reset_metric_from_dict(training_loss_dict_metric)
                epoch_loss.append(training_result["loss"])  # To get average at the end of epoch
                training_result["learning_rate"] = learning_rate_holder.result().numpy()[0]
                learning_rate_holder.reset_states()
                tepoch.set_postfix(**training_result)

                train_history[global_step] = training_result

                # Do after provided steps
                if validation_interval_steps:
                    if steps_covered % validation_interval_steps == 0:
                        val_result = do_validation(validation_dataset)
                        validation_history[global_step] = val_result
                        logging.info(pformat("Validation result before training {}".format(val_result)))

                # Save model
                if model_save_interval_steps:
                    if steps_covered % model_save_interval_steps == 0:
                        checkpoint_manager.save()

        # After an epoch
        checkpoint_manager.save()
        end_epoch_time = time.time()
        val_result = do_validation(validation_dataset)
        if val_result:
            logging.info(pformat("Validation result before training {}".format(val_result)))
        logging.info(
            pformt(
                "Epoch {}/{} --- Mean Loss {} --- Time {} seconds".format(
                    epoch + 1, epochs, tf.reduce_mean(epoch_loss), end_epoch_time - start_epoch_time
                )
            )
        )

    history["train_history"] = train_history
    history["validation_history"] = validation_history

    return history
