import tensorflow as tf
from absl import logging


def load_checkpoint_custom(model, checkpoint_dir=None, checkpoint_path=None, options=None, **kwargs):
    """[summary]

    Args:
        checkpoint_dir ([str]): [Location of the model]
    """
    try:
        options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    except:
        options = tf.CheckpointOptions(experimental_io_device="/job:localhost")

    if checkpoint_dir:
        if tf.io.gfile.exists(checkpoint_dir):
            if tf.io.gfile.isdir(checkpoint_dir) is False:
                raise ValueError("checkpoint_dir expects a directory not a file {}.".format(checkpoint_dir))
    if checkpoint_path:
        if tf.io.gfile.isdir(checkpoint_path) is True:
            raise ValueError("checkpoint_path expects a checkpoint-file not a directory {}.".format(checkpoint_path))
    checkpoint = tf.train.Checkpoint(model=model, **kwargs)
    if checkpoint_path is None and checkpoint_dir:
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        if checkpoint_dir:
            logging.info("No ❌❌ checkpoint found in {}".format(checkpoint_dir))
        else:
            logging.info("No ❌❌ checkpoint found")
        return None
    else:
        if options:
            status = checkpoint.restore(checkpoint_path, options=options)
        else:
            status = checkpoint.restore(checkpoint_path)
        # Important
        if status.assert_existing_objects_matched():
            logging.info("Successful ✅✅: Model checkpoints matched and loaded from {}".format(checkpoint_path))
            return checkpoint
        else:
            logging.info("Failed ❌❌ to load the checkpoint. Status Assertion Failed.")
    return None
