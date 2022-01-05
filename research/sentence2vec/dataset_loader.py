from random import shuffle

import tensorflow as tf
import tensorflow_text as tf_text

from tf_transformers.data import TFReader
from tf_transformers.utils import tf_utils


def split_fn(item, delimiter_pattern):
    text_splitted = tf_text.regex_split(item['text'][0], delimiter_pattern, keep_delim_regex_pattern=' ', name='split')
    return text_splitted


def filter_sentence_by_count(item, minimum_sentences):
    n_sentences = tf.cast(item.row_splits[1], tf.int32)
    if tf.less(n_sentences, minimum_sentences):
        return tf.constant(False)
    return tf.constant(True)


def filter_empty_string(sentences):
    """This will ensure, if any of the sentence in list of sentences is '' or' ', empty string,
    that will be filtered out"""
    sentences = sentences[0]
    valid_string_indexes = tf.squeeze(tf.where(tf.not_equal(tf.strings.length(sentences), 0)), axis=1)
    sentences = tf.gather(sentences, valid_string_indexes)

    # We need RaggedTensor
    sentences = tf.RaggedTensor.from_tensor(tf.expand_dims(sentences, axis=0))
    return sentences


def attention_mask_square(nd):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    dtype = tf_utils.get_dtype()
    ns = nd
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def prepare_centre_neighbour_sentences(item, window_length):

    n_sentences = tf.cast(item.row_splits[1], tf.int32)
    # 1 - n_sentences-1
    centre_index = tf.random.uniform(minval=1, maxval=n_sentences - 1, shape=(), dtype=tf.int32)

    min_window_boundary = tf.maximum(0, centre_index - window_length)
    max_window_boundary = tf.minimum(centre_index + window_length, n_sentences)

    neighbour_indexes = tf.squeeze(
        tf.where(tf.not_equal(tf.range(min_window_boundary, max_window_boundary), centre_index)), axis=1
    )

    # Randomly sample it for each example with different number of neighbours
    # n_neighbours = tf.random.uniform(minval=1, maxval = tf.shape(neighbour_indexes)[0], shape=(), dtype=tf.int32)
    # neighbour_index_ids = tf.random.uniform(minval=0, \
    # maxval = tf.shape(neighbour_indexes)[0], shape=(n_neighbours,), dtype=tf.int32)
    # neighbour_indexes = tf.gather(neighbour_indexes, neighbour_index_ids)

    centre_sentence = tf.gather(item[0], centre_index)
    neighbour_sentence = tf.gather(item[0], neighbour_indexes)
    # Repeat n-times
    n = tf.shape(neighbour_sentence)[0]
    centre_sentence = tf.repeat(centre_sentence, [n])

    # vec = tf.expand_dims(tf.range(batch_size), 0)

    # We use this same vector for all similiar sentences
    # Then we calculate scores (M x M transpose)
    # use this scores to generate mask for softmax
    # vec = tf.stop_gradient(tf.random.uniform(shape=(1, 32))) # keep a unique random
    # multiply = tf.stack([n, 1], axis=0)
    # tensor = tf.cast(tf.tile(vec, multiply), tf.float32)

    return {'centre_sentence': centre_sentence, 'neighbour_sentence': neighbour_sentence}


def shuffle_locally(item):
    indices = tf.range(start=0, limit=tf.shape(item['centre_sentence'])[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    centre_sentence = tf.gather(item['centre_sentence'], shuffled_indices)
    neighbour_sentence = tf.gather(item['neighbour_sentence'], shuffled_indices)
    # mask = tf.gather(item['mask'], shuffled_indices)

    result = {}
    result['centre_sentence'] = centre_sentence
    result['neighbour_sentence'] = neighbour_sentence
    # result['mask'] = mask

    return result


# def create_softmax_mask(item):
#     scores = tf.matmul(item['mask'], item['mask'], transpose_b=True)
#     # Replace similar scores place with -1e-9
#     scores_mask = tf.where(tf.equal(scores, tf.linalg.diag_part(scores)),\
# _large_compatible_negative(scores.dtype), tf.cast(0.0, scores.dtype))
#     # Reset only diagonal entries back to 1.0
#     scores_mask = tf.linalg.set_diag(scores_mask, tf.zeros(shape=(batch_size)), name='make_diagonal_one')
#     item['mask'] = scores_mask
#     return item


def text_to_features(item, tokenizer_layer, max_seq_length):
    """Convert item a tuple (src, target) into features"""
    result = {}
    input_ids = tokenizer_layer({'text': item['centre_sentence']})[: max_seq_length - 2]  # 2 for CLS and SEP
    # Add CLS and SEP
    input_ids = tf_text.combine_segments(
        [input_ids], start_of_sequence_id=tokenizer_layer.cls_token_id, end_of_segment_id=tokenizer_layer.sep_token_id
    )[0]
    input_ids, input_mask = tf_text.pad_model_inputs(input_ids, max_seq_length=max_seq_length)
    input_type_ids = tf.zeros_like(input_mask)

    result['centre_input_ids'] = input_ids
    result['centre_input_mask'] = input_mask
    result['centre_input_type_ids'] = input_type_ids

    input_ids = tokenizer_layer({'text': item['neighbour_sentence']})[: max_seq_length - 2]  # 2 for CLS and SEP
    # Add CLS and SEP
    input_ids = tf_text.combine_segments(
        [input_ids], start_of_sequence_id=tokenizer_layer.cls_token_id, end_of_segment_id=tokenizer_layer.sep_token_id
    )[0]
    input_ids, input_mask = tf_text.pad_model_inputs(input_ids, max_seq_length=max_seq_length)
    input_type_ids = tf.zeros_like(input_mask)

    result['neighbour_input_ids'] = input_ids
    result['neighbour_input_mask'] = input_mask
    result['neighbour_input_type_ids'] = input_type_ids

    # labels = {'mask': item['mask']}
    labels = {}

    return result, labels


# TODO : Dont Delete
# def text_to_features(item):
#     """Convert item a tuple (src, target) into features"""
#     result = {}
#     centre = tokenizer_layer({'text': item['centre_sentence']})
#     neighbour = tokenizer_layer({'text': item['neighbour_sentence']})

#     result = {}

#     result['centre_input_ids'] = centre['input_ids']
#     result['centre_input_mask'] = centre['input_mask']
#     result['centre_input_type_ids'] = centre['input_type_ids']

#     result['neighbour_input_ids'] = neighbour['input_ids']
#     result['neighbour_input_mask'] = neighbour['input_mask']
#     result['neighbour_input_type_ids'] = neighbour['input_type_ids']

#     labels = {}

#     return result, labels


def get_dataset(delimiter_pattern, minimum_sentences, window_length, tokenizer_layer, max_seq_length, batch_size):
    schema = {"text": ("var_len", "bytes")}
    all_files = tf.io.gfile.glob("gs://legacyai-bucket/c4/en/3.0.1/*.tfrecord*")
    shuffle(all_files)
    tf_reader = TFReader(schema=schema, tfrecord_files=all_files)
    dataset = tf_reader.read_record(keys=['text'])

    ds = dataset.map(lambda x: split_fn(x, delimiter_pattern), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(filter_empty_string, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x: filter_sentence_by_count(x, minimum_sentences))
    ds = ds.map(lambda x: prepare_centre_neighbour_sentences(x, window_length), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.unbatch()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(shuffle_locally, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.map(create_softmax_mask, num_parallel_calls = tf.data.AUTOTUNE)
    ds = ds.map(lambda x: text_to_features(x, tokenizer_layer, max_seq_length), num_parallel_calls=tf.data.AUTOTUNE)

    return ds
