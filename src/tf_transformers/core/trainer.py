import time

import tensorflow as tf
from absl import logging
from tqdm import tqdm

logging.set_verbosity("INFO")


def SimpleTrainer(model, optimizer, loss_fn, dataset, epochs, num_train_examples, batch_size, steps_per_call=100):
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
