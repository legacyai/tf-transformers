import time

import tensorflow as tf
from absl import logging
from tqdm import tqdm

logging.set_verbosity("INFO")

# Gradient Acculation credit goes to
# Goran Sandstorm Nysater


def SimpleTrainer(
    model,
    optimizer,
    loss_fn,
    dataset,
    epochs,
    num_train_examples,
    batch_size,
    steps_per_call=100,
    gradient_accumulation_steps=1,
):
    """Simple trainer / Gradient accumulation steps

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

        def reduce_grads(grads):
            acc_grads = []
            for var_grads in grads:
                acc_grads.append(tf.reduce_mean(var_grads, axis=0))
            return acc_grads

        def train_step(iterator):
            grads = [tf.TensorArray(tf.float32, size=steps) for v in model.trainable_variables]
            for i in tf.range(tf.convert_to_tensor(gradient_accumulation_steps)):
                batch_inputs, batch_labels = next(iterator)
                with tf.GradientTape() as tape:
                    model_outputs = model(batch_inputs)
                    loss = loss_fn(batch_labels, model_outputs)
                # training_loss.update_state(loss * strategy.num_replicas_in_sync)
                training_loss.update_state(loss)

                grads = tape.gradient(loss, model.trainable_variables)
                for j, g in enumerate(grads):
                    grads[j].write(i, g)
            grads_reduced = reduce_grads([grad_arr.stack() for grad_arr in grads])
            optimizer.apply_gradients(zip(grads_reduced, model.trainable_variables))
            # Account optimizer for global_batch steps
            optimizer.iterations.assign_add(gradient_accumulation_steps - 1)

        for _ in tf.range(tf.convert_to_tensor(steps_per_call)):
            train_step(batch_inputs, batch_labels)

    # Make dataset iterator

    # To make it properly divisible
    steps_per_call = steps_per_call // gradient_accumulation_steps
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
