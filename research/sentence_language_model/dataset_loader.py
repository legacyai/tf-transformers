import os
from random import shuffle

import tensorflow as tf


def get_dataset(data_directory, masked_lm_map_fn, batch_size):
    """Convert text to tf.data.Dataset after map fn

    Args:
        data_directory ([type]): [description]
        masked_lm_map_fn ([type]): [description]
        batch_size ([type]): [description]

    Returns:
        [type]: [description]
    """

    def filter_out_empty_mask(x, y):
        """When an example doesn't have multiple sentences\
            there wont be any masked sentence. Ignore those examples,
            as nothing to predict.
            """
        return tf.greater(tf.reduce_sum(tf.cast(tf.not_equal(x['masked_lm_positions'], 0), tf.int32)), 0)

    all_text_files = tf.io.gfile.glob(os.path.join(data_directory, '*.txt'))
    shuffle(all_text_files)
    ds = tf.data.TextLineDataset(all_text_files)

    # We need to add the text as dict
    ds = ds.map(lambda x: {'text': x}, num_parallel_calls=tf.data.AUTOTUNE)

    # Do MLM
    ds = ds.map(masked_lm_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter examples if there is not atleast single MASK sentence
    ds = ds.filter(filter_out_empty_mask)

    # Batch
    ds = ds.batch(batch_size, drop_remainder=True)

    # Shuffle and Prefetch
    ds = ds.shuffle(100, reshuffle_each_iteration=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Auto SHARD
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    ds = ds.with_options(options)

    return ds
