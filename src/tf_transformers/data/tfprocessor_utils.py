import tensorflow as tf
from absl import logging

from tf_transformers.data import pad_ragged, separate_x_y

logging.set_verbosity("INFO")


class TFProcessor(object):
    """
    TFProcessor class . This class is responsible to read data, \
    and convert it to a tf.data.Dataset

    """

    def auto_batch(
        self,
        tf_dataset,
        batch_size,
        x_keys=None,
        y_keys=None,
        shuffle=False,
        drop_remainder=False,
        shuffle_buffer_size=10000,
    ):
        """Auto batching

        Args:
            tf_dataset : TF dataset
            batch_size : batch size
            x_keys (optional): List of key names. We will filter based on this.
            y_keys (optional): List of key names.
            shuffle (bool, optional): [description]. Defaults to False.
            drop_remainder (bool, optional): [description]. Defaults to False.
            shuffle_buffer_size (int, optional): [description]. Defaults to 10000.

        Returns:
            batched tf dataset
        """
        # element_spec = tf_dataset.element_spec
        dataset = tf_dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.map(pad_ragged, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # fmt: off
        if x_keys and y_keys:
            dataset = dataset.map(lambda x: separate_x_y(x, x_keys, y_keys), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # noqa
        # fmt: on
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=None, reshuffle_each_iteration=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def process(self, parse_fn, verbose=10000):
        """This function will iterate over parse_fn and keep writing it TFRecord
        Args:

            parse_fn: function which should be an iterator or generator
        """
        data = {}
        if hasattr(parse_fn, "__iter__") and not hasattr(parse_fn, "__len__"):
            counter = 0
            for entry in parse_fn:
                for k, v in entry.items():
                    if k in data:
                        data[k].append(v)
                    else:
                        data[k] = [v]
                counter += 1

                if counter % verbose == 0:
                    logging.info("Processed  {} examples so far".format(counter))

            logging.info("Total individual observations/examples written is {}".format(counter))
            data_ragged = {k: tf.ragged.constant(v) for k, v in data.items()}
            dataset = tf.data.Dataset.from_tensor_slices(data_ragged)
            return dataset
        else:
            raise ValueError("Expected `parse_fn` to be a generator/iterator ")
