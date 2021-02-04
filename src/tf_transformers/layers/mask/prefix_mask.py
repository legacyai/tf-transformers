import tensorflow as tf


def attention_mask_square(nd, *, dtype=tf.float32):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    ns = nd
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def prefix_mask(mask):
    """
    This function has to be called on mask which is 1-D tensor of values either 0 or 1 or both

    text = ['Which' , 'place', 'you"re' , 'from?', 'Iam', 'in', 'Kerala']
    mask = [[1      ,   1    ,    1     ,    1   ,   0  ,   0 ,     0]]

    input_ones_mask ( (n_input_words + n_output_words) x  (n_output_words) )

            Which   place  you"re  from?
      Which [[1.,   1.,     1.,     1.],
      place  [1.,   1.,     1.,     1.],
      you"re [1.,   1.,     1.,     1.],
      from?  [1.,   1.,     1.,     1.],
      Iam    [1.,   1.,     1.,     1.],
      in     [1.,   1.,     1.,     1.],
      Kerala [1.,   1.,     1.,     1.]]

    input_to_output_zero_mask (n_input_words, n_output_words)
             Iam     in   Kerala
      Which  [[0.,   0.,    0.],
      place  [0.,    0.,    0.],
      you"re [0.,    0.,    0.],
      from   [0.,    0.,    0.]]

    output_upper_triangular_mask (n_output_words x n_output_words)
            Iam     in    Kerala
      Iam   [[1.,   0.,     0.],
      in     [1.,   1.,     0.],
      Kerala [1.,   1.,     1.]]

            Which   place  you"re  from?  Iam in  Kerala
      Which [[1.,   1.,     1.,     1.     0,  0,   0],
      place  [1.,   1.,     1.,     1.     0,  0,   0],
      you"re [1.,   1.,     1.,     1.     0,  0,   0],
      from?  [1.,   1.,     1.,     1.     0,  0,   0],
      Iam    [1.,   1.,     1.,     1.     1., 0,   0],
      in     [1.,   1.,     1.,     1.     1., 1.,  0],
      Kerala [1.,   1.,     1.,     1.     1., 1.,  1.]]

    Finally merge (input_ones_mask and [input_to_output_zero_mask, output_upper_triangular_mask])
    """
    mask = tf.cast(mask, tf.float32)
    n_input_words = tf.reduce_sum(mask)  # total no of ones
    n_output_words = tf.reduce_sum(tf.cast(tf.equal(mask, 0), tf.float32))  # total number of zeros

    # input_to_input and output_to_input_words_mask
    input_ones_mask = tf.ones((n_input_words + n_output_words, n_input_words))
    # input_to_output_words
    input_to_output_zero_mask = tf.zeros((n_input_words, n_output_words))
    # Upper triangular mask , output_words should never peek into future
    output_upper_triangular_mask = attention_mask_square(n_output_words)

    to_outputs = tf.concat([input_to_output_zero_mask, output_upper_triangular_mask], axis=0)
    final = tf.concat([input_ones_mask, to_outputs], axis=1)
    return final
