import datasets
import tensorflow as tf
import tensorflow_text as tf_text

dataset = datasets.load_dataset("scientific_papers", "pubmed")


def load_dataset(tokenizer_layer, encoder_seq_length, decoder_seq_length, batch_size):
    """
    Return dataset using tensorflow tokenizer tftext to in-batch processing.


    """

    decoder_start_token_id = tokenizer_layer.pad_token_id
    eos_token_id = tokenizer_layer.eos_token_id

    def text_to_features(item):
        """Encode text to features"""

        inputs = {}
        labels = {}

        encoder_input_ids = tokenizer_layer({'text': item['article']})
        encoder_input_mask = tf.ones_like(encoder_input_ids)
        encoder_input_ids, encoder_input_mask = tf_text.pad_model_inputs(
            encoder_input_ids, max_seq_length=encoder_seq_length
        )

        inputs['encoder_input_ids'] = encoder_input_ids
        inputs['encoder_input_mask'] = encoder_input_mask

        decoder_input_ids = tokenizer_layer({'text': item['abstract']})
        decoder_input_ids = decoder_input_ids[:, :-1]  # Exclude special token (eos_token_id) we add it explicitly
        decoder_input_ids = decoder_input_ids[:, : decoder_seq_length - 1]  # Fos decoder_start_token_id, eos_token_id
        decoder_input_ids = tf_text.combine_segments(
            [decoder_input_ids], start_of_sequence_id=decoder_start_token_id, end_of_segment_id=eos_token_id
        )[0]
        decoder_input_ids, input_mask = tf_text.pad_model_inputs(
            decoder_input_ids, max_seq_length=decoder_seq_length + 1
        )

        inputs['decoder_input_ids'] = decoder_input_ids[:, :-1]  # Exclude last token

        labels['labels'] = decoder_input_ids[:, 1:]  # not including first word
        labels['labels_mask'] = input_mask[:, 1:]

        return inputs, labels

    ds_train = dataset['train'].to_dict()
    ds_train = tf.data.Dataset.from_tensor_slices(ds_train).shuffle(1024).batch(batch_size)
    ds_train = ds_train.map(text_to_features, num_parallel_calls=tf.data.AUTOTUNE)
    return ds_train, dataset['train'].num_rows


def load_dataset_eval(tokenizer_layer, encoder_seq_length, decoder_seq_length, batch_size):
    """
    Return dataset using tensorflow tokenizer tftext to in-batch processing.


    """

    decoder_start_token_id = tokenizer_layer.pad_token_id
    eos_token_id = tokenizer_layer.eos_token_id

    def text_to_features(item):
        """Encode text to features"""

        inputs = {}
        labels = {}

        encoder_input_ids = tokenizer_layer({'text': item['article']})
        encoder_input_mask = tf.ones_like(encoder_input_ids)
        encoder_input_ids, encoder_input_mask = tf_text.pad_model_inputs(
            encoder_input_ids, max_seq_length=encoder_seq_length
        )

        inputs['encoder_input_ids'] = encoder_input_ids
        inputs['encoder_input_mask'] = encoder_input_mask

        decoder_input_ids = tokenizer_layer({'text': item['abstract']})
        decoder_input_ids = decoder_input_ids[:, :-1]  # Exclude special token (eos_token_id) we add it explicitly
        decoder_input_ids = decoder_input_ids[:, : decoder_seq_length - 1]  # Fos decoder_start_token_id, eos_token_id
        decoder_input_ids = tf_text.combine_segments(
            [decoder_input_ids], start_of_sequence_id=decoder_start_token_id, end_of_segment_id=eos_token_id
        )[0]
        decoder_input_ids, input_mask = tf_text.pad_model_inputs(
            decoder_input_ids, max_seq_length=decoder_seq_length + 1
        )

        inputs['decoder_input_ids'] = decoder_input_ids[:, :-1]  # Exclude last token

        labels['labels'] = decoder_input_ids[:, 1:]  # not including first word
        labels['labels_mask'] = input_mask[:, 1:]
        labels['text'] = item['abstract']
        return inputs, labels

    ds_eval = dataset['validation'].to_dict()
    ds_eval = tf.data.Dataset.from_tensor_slices(ds_eval).batch(batch_size, drop_remainder=True)
    ds_eval = ds_eval.map(text_to_features, num_parallel_calls=tf.data.AUTOTUNE)
    return ds_eval, dataset['validation'].num_rows
