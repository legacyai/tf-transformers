import tensorflow as tf
import time


def trainer(
    model,
    batch_size,
    dataset,
    total_examples,
    loss_fn,
    optimizer,
    epochs,
    validation_dataset,
    validation_loss_fn,
    model_checkpoint_dir,
    steps_per_epoch=None,
    model_save_interval_epochs=1,
    max_number_of_models=10,
    model_save_interval_steps=1,
    overwrite_checkpoint_dir=False,
    validation_interval_steps=None,
    train_steps=None,
    steps_per_call=100,
    eval_callback=None,
):

    total_batches = total_examples // batch_size
    if steps_per_epoch is None:
        steps_per_epoch = total_examples
        n_repeats = epochs
    else:
        if steps_per_epoch > total_batches:
            n_repeats = steps_per_epoch // total_batches

    # Model checkpoint
    if not overwrite_checkpoint_dir:
        import os

        if os.path.exists(model_checkpoint_dir):
            raise FileExistsError("Model directory exists")

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_checkpoint_dir, max_to_keep=max_number_of_models)

    # Train Functions
    @tf.function
    def train(iterator):
        """The step function for one training step"""

        def train_step(batch_inputs, batch_labels):
            """The computation to run on each TPU device."""
            with tf.GradientTape() as tape:
                model_outputs = model(batch_inputs)
                loss = loss_fn(batch_labels, model_outputs)

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
    def _validate(iterator):
        """Validation step"""
        for (batch_inputs, batch_labels) in iterator:
            model_outputs = model(batch_inputs)
            loss = validation_loss_fn(batch_labels, model_outputs)
            for name, loss_value in loss.items():
                validation_loss = validation_loss_dict_metric[name]
                validation_loss.update_state(loss_value)

    # do validation
    def do_validation(validation_dataset):
        validation_result = {}
        if validation_dataset and validation_loss_fn:
            _validate(validation_dataset)
        loss = validation_loss.result()
        validation_result = {name: metric.result() for name, metric in validation_loss_dict_metric.items()}
        for name, metric in validation_loss_dict_metric.items():
            metric.reset_states()
        if eval_callback:
            score = eval_callback(kwargs)
            validation_result["val_score"] = score
        return validation_result

    # Metrics
    learning_rate_holder = tf.keras.metrics.Mean(
        "learning_rate_holder", dtype=tf.float32
    )  # We store learning rate here and reset after every global steps

    # Sample Call
    for (batch_inputs, batch_labels) in dataset.take(1):
        pass
    model_outputs = model(batch_inputs)
    train_loss_dict = loss_fn(batch_labels, model_outputs)
    training_loss_dict_metric = {name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in train_loss_dict}

    if validation_dataset and validation_loss_fn:
        for (batch_inputs, batch_labels) in dataset.take(1):
            pass
        model_outputs = model(batch_inputs)
        valid_loss_dict = validation_loss_fn(batch_labels, model_outputs)
        validation_loss_dict_metric = {name: tf.keras.metrics.Mean(name, dtype=tf.float32) for name in valid_loss_dict}

    # dataset to iterator
    dataset_iterator = iter(dataset.repeat(n_repeats + 1))
    training_loss_holder = []
    learning_rate_holder_history = []
    validation_loss_holder = []
    validation_score = []
    validation_steps = []

    # Do before model got trained
    if validation_dataset:
        val_result = do_validation(validation_dataset)
        validation_loss_holder.append(val_result["val_loss"])
        validation_score.append(val_result["val_score"])
        validation_steps.append(0)

    # Main Loop
    history = {}
    for epoch in range(epochs):
        epoch_loss = []
        start_epoch_time = time.time()
        for step in range(steps_per_epoch // steps_per_call):

            steps_covered = (step + 1) * steps_per_call
            start_time = time.time()
            train(dataset_iterator)
            end_time = time.time()

            training_result = {name: metric.result() for name, metric in training_loss_dict_metric.items()}
            training_print = ["{}: {}".format(k, v) for k, v in training_result.items()]
            for name, metric in training_loss_dict_metric.items():
                metric.reset_states()
            epoch_loss.append(training_result["loss"])
            learning_rate_holder_history.append(learning_rate_holder.result())
            learning_rate_holder.reset_states()
            print(
                "Epoch {} --- Step {}/{} --- LR --- {} Loss {} --- Time {} seconds ".format(
                    epoch + 1,
                    steps_covered,
                    steps_per_epoch,
                    learning_rate_holder_history[-1],
                    training_print,
                    end_time - start_time,
                ),
                end="\r",
            )
            # Do after provided steps
            if validation_interval_steps:
                if steps_covered % validation_interval_steps == 0:
                    start_time = time.time()
                    val_result = do_validation(validation_dataset)
                    end_time = time.time()
                    validation_loss_holder.append(val_result["loss"])
                    validation_score.append(val_result["val_score"])
                    validation_steps.append(steps_covered)
                    print(
                        "Epoch {} --- validation Step {} --- Loss {} --- eval score {} Time {} seconds ".format(
                            epoch,
                            steps_covered,
                            val_result,
                            validation_score[-1],
                            end_time - start_time,
                        ),
                        end="\r",
                    )
                    manager.save()
            training_loss_holder.extend(epoch_loss)
            end_epoch_time = time.time()
            print(
                "Epoch {}/{} --- Mean Loss {} in {} seconds".format(
                    epoch + 1, epochs, tf.reduce_mean(epoch_loss), end_epoch_time - start_epoch_time
                )
            )
        # Do after every epoch
        if validation_dataset:
            val_result = do_validation(validation_dataset)
            validation_loss_holder.append(val_result["val_loss"])
            validation_score.append(val_result["val_score"])
            validation_steps.append(steps_covered)
        manager.save()

    history["training_loss"] = training_loss_holder
    history["val_loss"] = validation_loss_holder
    history["val_score"] = validation_score
    history["val_steps"] = validation_steps
    history["learning_rate"] = learning_rate_holder_history

    return history
