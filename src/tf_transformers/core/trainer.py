import time

import tensorflow as tf
from absl import logging
from tqdm import tqdm

logging.set_verbosity("INFO")


def Trainer(model, optimizer, loss_fn, dataset, epochs, num_train_examples, batch_size, steps_per_call=100):
    """Simple trainer

    Args:
        model ([type]): [description]
        optimizer ([type]): [description]
        loss ([type]): [description]
        dataset ([type]): [description]
        epochs ([type]): [description]
        num_train_examples ([type]): [description]
        batch_size ([type]): [description]
        steps_per_call (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """

    @tf.function
    def train(iterator):
        """The step function for one training step"""

        def train_step(batch_inputs, batch_labels):
            """The computation to run on each TPU device."""
            with tf.GradientTape() as tape:
                model_outputs = model(batch_inputs)
                loss = loss_fn(batch_labels, model_outputs)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            # training_loss.update_state(loss * strategy.num_replicas_in_sync)
            training_loss.update_state(loss)
            current_lr = optimizer._decayed_lr(tf.float32)
            learning_rate_holder.update_state(current_lr)

        for _ in tf.range(tf.convert_to_tensor(steps_per_call)):
            batch_inputs, batch_labels = next(iterator)
            train_step(batch_inputs, batch_labels)

    # Make dataset iterator

    GLOBAL_STEP = epochs * (num_train_examples // (batch_size * steps_per_call))
    logging.info("Global Steps {}".format(GLOBAL_STEP))
    iter_dataset = iter(dataset)
    training_loss = tf.keras.metrics.Mean(
        "training_loss", dtype=tf.float32
    )  # We store loss here and reset after every global steps
    learning_rate_holder = tf.keras.metrics.Mean(
        "learning_rate_holder", dtype=tf.float32
    )  # We store loss here and reset after every global steps
    loss_holder = []
    learning_rate_holder_history = []
    for step_iter in tqdm(range(GLOBAL_STEP)):
        start_time = time.time()
        train(iter_dataset)
        end_time = time.time()
        loss_holder.append(training_loss.result())
        learning_rate_holder_history.append(learning_rate_holder.result())
        logging.info(
            "Global step {}, time {} seconds, loss {}, learning_rate {}".format(
                step_iter, round(end_time - start_time, 4), loss_holder[-1], learning_rate_holder_history[-1]
            )
        )
        training_loss.reset_states()

    return loss_holder


def SimpleTrainerGradientAccumulation(
    model,
    optimizer,
    loss_fn,
    dataset,
    epochs,
    num_train_examples,
    batch_size,
    steps_per_call=100,
    gradient_accumulation_steps=8,
):
    """Simple trainer

    Args:
        model ([type]): [description]
        optimizer ([type]): [description]
        loss ([type]): [description]
        dataset ([type]): [description]
        epochs ([type]): [description]
        num_train_examples ([type]): [description]
        batch_size ([type]): [description]
        steps_per_call (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """

    @tf.function
    def train(iterator):
        """The step function for one training step"""

        def get_gradient(inputs):
            batch_inputs, batch_labels = inputs
            with tf.GradientTape() as tape:
                model_outputs = model(batch_inputs)
                loss = loss_fn(batch_labels, model_outputs)

            grads = tape.gradient(loss, model.variables)
            return grads, loss

        def train_step(iterator):
            """The computation to run on each TPU device."""

            # Step 0
            inputs = next(iterator)
            grads, loss = get_gradient(inputs)
            training_loss.update_state(loss)
            grads_holder = [
                tf.TensorArray(tf.float32, size=1, dynamic_size=True, infer_shape=False, element_shape=v.shape)
                for v in model.trainable_variables
            ]
            for (j, g) in enumerate(grads):
                if g is not None:
                    grads_holder[j] = grads_holder[j].write(0, g)
                else:
                    grads_holder[j] = grads_holder[j].write(0, tf.zeros_like(model.trainable_variables[j]))

            # Further iterations
            for i in range(gradient_accumulation_steps - 1):
                inputs = next(iterator)
                grads, loss = get_gradient(inputs)
                training_loss.update_state(loss)
                for (j, g) in enumerate(grads):
                    if g is not None:
                        new_tensor = grads_holder[j].read(0) + g
                        grads_holder[j] = grads_holder[j].write(0, new_tensor)

            # Unstack grads holder
            grads_holder = [
                tf.squeeze(grad_arr.stack(), axis=0) / gradient_accumulation_steps for grad_arr in grads_holder
            ]
            optimizer.apply_gradients(zip(grads_holder, model.trainable_variables))
            optimizer.iterations.assign_add(gradient_accumulation_steps - 1)
            # training_loss.update_state(loss * strategy.num_replicas_in_sync)

        for _ in tf.range(tf.convert_to_tensor(steps_per_call)):
            train_step(iterator)

    # Make dataset iterator

    GLOBAL_STEP = epochs * (num_train_examples // (batch_size * steps_per_call))
    if GLOBAL_STEP <= 0:
        GLOBAL_STEP = 1
    logging.info("Global Steps {}".format(GLOBAL_STEP))
    steps_per_call = steps_per_call // gradient_accumulation_steps
    iter_dataset = iter(dataset)
    training_loss = tf.keras.metrics.Mean(
        "training_loss", dtype=tf.float32
    )  # We store loss here and reset after every global steps

    loss_holder = []
    for step_iter in tqdm(range(GLOBAL_STEP)):
        start_time = time.time()
        train(iter_dataset)
        end_time = time.time()
        loss_holder.append(training_loss.result())
        logging.info(
            "Global step {}, time {} seconds, loss {}".format(
                step_iter, round(end_time - start_time, 4), loss_holder[-1]
            )
        )
        training_loss.reset_states()

    return loss_holder


def SimpleTrainer(
    model,
    optimizer,
    loss_fn,
    dataset,
    epochs,
    num_train_examples,
    batch_size,
    steps_per_call=100,
    gradient_accumulation_steps=None,
):
    if gradient_accumulation_steps:
        loss_holder = SimpleTrainerGradientAccumulation(
            model,
            optimizer,
            loss_fn,
            dataset,
            epochs,
            num_train_examples,
            batch_size,
            steps_per_call,
            gradient_accumulation_steps,
        )
        return loss_holder
    else:
        loss_holder = Trainer(
            model, optimizer, loss_fn, dataset, epochs, num_train_examples, batch_size, steps_per_call
        )
        return loss_holder
