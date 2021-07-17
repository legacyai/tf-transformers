import glob
import json

import tensorflow as tf
import tensorflow_text as tf_text
from transformers import AlbertTokenizer

from tf_transformers.core import TPUTrainer
from tf_transformers.data import TFReader
from tf_transformers.losses import cross_entropy_loss
from tf_transformers.optimization import create_optimizer


def dynamic_masking_from_features(
    max_seq_len, max_predictions_per_batch, vocab_size, cls_id, sep_id, unk_id, pad_id, mask_id
):

    """Dynamic Masking from input_ids (saved as tfrecord)"""
    # Truncate inputs to a maximum length.
    trimmer = tf_text.RoundRobinTrimmer(max_seq_length=max_seq_len)

    # Random Selector
    random_selector = tf_text.RandomItemSelector(
        max_selections_per_batch=max_predictions_per_batch,
        selection_rate=0.2,
        unselectable_ids=[cls_id, sep_id, unk_id, pad_id],
    )

    # Mask Value chooser (Encapsulates the BERT MLM token selection logic)
    mask_values_chooser = tf_text.MaskValuesChooser(vocab_size, mask_id, 0.8)

    def map_mlm(item):

        segments = item['input_ids']
        trimmed_segments = trimmer.trim([segments])

        # We replace trimmer with slice [:_MAX_SEQ_LEN-2] operation # 2 to add CLS and SEP
        # input_ids = item['input_ids'][:_MAX_SEQ_LEN-2]

        # Combine segments, get segment ids and add special tokens.
        segments_combined, segment_ids = tf_text.combine_segments(
            trimmed_segments, start_of_sequence_id=cls_id, end_of_segment_id=sep_id
        )

        # We replace segment with concat
        # input_ids = tf.concat([[_START_TOKEN], input_ids, [_END_TOKEN]], axis=0)

        # Apply dynamic masking
        masked_token_ids, masked_pos, masked_lm_ids = tf_text.mask_language_model(
            segments_combined, item_selector=random_selector, mask_values_chooser=mask_values_chooser
        )

        # Prepare and pad combined segment inputs
        input_word_ids, input_mask = tf_text.pad_model_inputs(masked_token_ids, max_seq_length=max_seq_len)
        input_type_ids, _ = tf_text.pad_model_inputs(segment_ids, max_seq_length=max_seq_len)

        # Prepare and pad masking task inputs
        # Masked lm weights will mask the weights
        masked_lm_positions, masked_lm_weights = tf_text.pad_model_inputs(
            masked_pos, max_seq_length=max_predictions_per_batch
        )
        masked_lm_ids, _ = tf_text.pad_model_inputs(masked_lm_ids, max_seq_length=max_predictions_per_batch)

        inputs = {}
        inputs['input_ids'] = input_word_ids
        inputs['input_type_ids'] = input_type_ids
        inputs['input_mask'] = input_mask
        inputs['masked_lm_positions'] = masked_lm_positions

        labels = {}
        labels['masked_lm_labels'] = masked_lm_ids
        labels['masked_lm_weights'] = masked_lm_weights  # Mask

        return (inputs, labels)

    return map_mlm


def get_tfdataset_from_tfrecords(tfrecord_path_list):
    """Get tf dataset from tfrecords"""
    all_files = []
    for tfrecord_path in tfrecord_path_list:
        all_files.extend(glob.glob("{}/*.tfrecord".format(tfrecord_path)))
    schema = json.load(open("{}/schema.json".format(tfrecord_path)))
    tf_reader = TFReader(schema=schema, tfrecord_files=all_files)
    train_dataset = tf_reader.read_record()
    return train_dataset


def filter_by_length(x, min_sen_len):
    """Filter by minimum sentence length (subwords)"""
    return tf.squeeze(tf.greater_equal(tf.shape(x['input_ids']), tf.constant(min_sen_len)), axis=0)


def filter_by_batch(x, y, batch_size):
    """Filter by batch size"""
    x_batch = tf.shape(x['input_ids'])[0]
    return tf.equal(x_batch, tf.constant(batch_size))


def get_model():
    """Model"""

    config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "intermediate_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "embedding_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "attention_head_size": 64,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": tokenizer.vocab_size,
        "layer_norm_epsilon": 1e-12,
    }

    from tf_transformers.models import BertModel

    model = BertModel.from_config(
        config,
        batch_size=BATCH_SIZE,
        use_masked_lm_positions=True,  # Add batch_size to avoid dynamic shapes
        return_all_layer_outputs=True,
    )

    return model


def get_optimizer():
    """Optimizer"""
    LEARNING_RATE = 1e-04
    NUM_TRAIN_STEPS = 200000
    NUM_WARMUP_STEPS = 30000
    OPTIMIZER_TYPE = "lamb"
    optimizer, learning_rate_fn = create_optimizer(
        init_lr=LEARNING_RATE,
        num_train_steps=NUM_TRAIN_STEPS,
        num_warmup_steps=NUM_WARMUP_STEPS,
        optimizer_type=OPTIMIZER_TYPE,
    )
    return optimizer


def lm_loss(y_true_dict, y_pred_dict):
    """Joint loss over all layers"""
    loss_dict = {}
    loss_holder = []
    for layer_count, per_layer_output in enumerate(y_pred_dict['all_layer_token_logits']):

        loss = cross_entropy_loss(
            labels=y_true_dict['masked_lm_labels'],
            logits=per_layer_output,
            label_weights=y_true_dict['masked_lm_weights'],
        )
        loss_dict['loss_{}'.format(layer_count + 1)] = loss
        loss_holder.append(loss)
    loss_dict['loss'] = tf.reduce_mean(loss_holder, axis=0)
    return loss_dict


# Callbacks


class MLMCallback:
    """Simple MLM Callback to check progress of the training"""

    def __init__(self, tokenizer, validation_sentences, top_k=10):
        """Init"""
        self.tokenizer = tokenizer
        self.validation_sentences = validation_sentences
        self.top_k = top_k

    def get_inputs(self):
        """Text to features"""
        inputs = self.tokenizer(self.validation_sentences, padding=True, return_tensors="tf")
        inputs_tf = {}
        inputs_tf["input_ids"] = inputs["input_ids"]
        inputs_tf["input_type_ids"] = inputs["token_type_ids"]
        inputs_tf["input_mask"] = inputs["attention_mask"]

        seq_length = tf.shape(inputs_tf['input_ids'])[1]
        inputs_tf['masked_lm_positions'] = tf.zeros_like(inputs_tf["input_ids"]) + tf.range(seq_length)

        return inputs_tf

    def __call__(self, trainer_params):
        """Main Call"""
        model = trainer_params['model']
        inputs_tf = self.get_inputs()
        outputs_tf = model(inputs_tf)

        # Get masked positions from each sentence
        masked_positions = tf.argmax(tf.equal(inputs_tf["input_ids"], self.tokenizer.mask_token_id), axis=1)
        for layer_count, layer_logits in enumerate(outputs_tf['all_layer_token_logits']):
            print("Layer {}".format(layer_count + 1))
            print("-------------------------------------------------------------------")
            for i, logits in enumerate(layer_logits):
                mask_token_logits = logits[masked_positions[i]]
                # 0 for probs and 1 for indexes from tf.nn.top_k
                top_words = tokenizer.decode(tf.nn.top_k(mask_token_logits, k=self.top_k)[1].numpy())
                print("Input ----> {}".format(validation_sentences[i]))
                print("Predicted words ----> {}".format(top_words.split()))
                print()


#### Define Constants

MAX_SEQ_LEN = 128
MAX_PREDICTIONS_PER_BATCH = 20
BATCH_SIZE = 512

TFRECORDS_PATH = ['home/sidhu/Datasets/TFRECORD_BOOKCORPUS', '/home/sidhu/Datasets/TFRECORD_WIKI']
TPU_ADDRESS = 'local'
DTYPE = 'bf16'

MODEL_DIR = 'bert_joint'
EPOCHS = 3
STEPS_PER_EPOCH = 50000
CALLBACK_STEPS = 5000
TRAINING_LOSS_NAMES = [
    'loss_1',
    'loss_2',
    'loss_3',
    'loss_4',
    'loss_5',
    'loss_6',
    'loss_7',
    'loss_8',
    'loss_9',
    'loss_10',
    'loss_11',
    'loss_12',
]

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
validation_sentences = [
    'Read the rest of this [MASK] to understand things in more detail.',
    'I want to buy the [MASK] because it is so cheap.',
    'The [MASK] was amazing.',
    'Sachin Tendulkar is one of the [MASK] palyers in the world.',
    '[MASK] is the capital of France.',
    'Machine Learning requires [MASK]',
    'He is working as a [MASK]',
    'She is working as a [MASK]',
]
mlm_callback = MLMCallback(tokenizer, validation_sentences)

# Prepare TF dataset

dynamic_mlm_fn = dynamic_masking_from_features(
    MAX_SEQ_LEN,
    MAX_PREDICTIONS_PER_BATCH,
    tokenizer.vocab_size,
    tokenizer.cls_token_id,
    tokenizer.sep_token_id,
    tokenizer.unk_token_id,
    tokenizer.pad_token_id,
    tokenizer.mask_token_id,
)

train_dataset = get_tfdataset_from_tfrecords(TFRECORDS_PATH)
train_dataset = train_dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=BATCH_SIZE))
train_dataset = train_dataset.map(dynamic_mlm_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.filter(lambda x, y: filter_by_batch(x, y, BATCH_SIZE))
train_dataset = train_dataset.shuffle(100)
train_dataset = train_dataset.prefetch(100)


# Call Trainer
trainer = TPUTrainer(tpu_address=TPU_ADDRESS, dtype=DTYPE)

# Run the training

trainer.run(
    model_fn=get_model,
    optimizer_fn=get_optimizer,
    train_dataset=train_dataset,
    train_loss_fn=lm_loss,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    model_checkpoint_dir=MODEL_DIR,  # gs://tft_free/
    batch_size=BATCH_SIZE,
    training_loss_names=TRAINING_LOSS_NAMES,
    validation_loss_names=None,
    validation_dataset=None,
    validation_loss_fn=None,
    validation_interval_steps=None,
    steps_per_call=100,
    enable_xla=False,
    callbacks=[mlm_callback],
    callbacks_interval_steps=[CALLBACK_STEPS],
    overwrite_checkpoint_dir=True,
    max_number_of_models=10,
    model_save_interval_steps=None,
    repeat_dataset=True,
)


import os

os.system("gsutil -m cp -R {} gs://tft_free/{}".format(MODEL_DIR, MODEL_DIR))
