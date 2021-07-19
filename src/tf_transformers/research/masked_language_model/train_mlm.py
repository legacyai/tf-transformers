import glob
import json

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from transformers import T5Tokenizer

from tf_transformers.core import TPUTrainer
from tf_transformers.data import TFReader
from tf_transformers.data.callbacks.mlm_callback import MLMCallback
from tf_transformers.data.processors.mlm import dynamic_masking_from_features
from tf_transformers.losses import cross_entropy_loss
from tf_transformers.optimization import create_optimizer
from tf_transformers.text import SentencepieceTokenizer


def load_tokenizer(cfg):
    """Load tf text based tokenizer"""
    model_file_path = cfg.tokenizer.model_file_path
    do_lower_case = cfg.tokenizer.do_lower_case
    special_tokens = cfg.tokenizer.special_tokens

    tokenizer_layer = SentencepieceTokenizer(
        model_file_path=model_file_path, lower_case=do_lower_case, special_tokens=special_tokens
    )

    return tokenizer_layer


def get_tfdataset_from_tfrecords(tfrecord_path_list):
    """Get tf dataset from tfrecords"""
    all_files = []
    for tfrecord_path in tfrecord_path_list:
        all_files.extend(glob.glob("{}/*.tfrecord".format(tfrecord_path)))
    schema = json.load(open("{}/schema.json".format(tfrecord_path)))
    tf_reader = TFReader(schema=schema, tfrecord_files=all_files)
    train_dataset = tf_reader.read_record()
    return train_dataset


def get_dataset(
    tfrecord_path_list,
    max_seq_len,
    max_predictions_per_batch,
    vocab_size,
    cls_token_id,
    sep_token_id,
    unk_token_id,
    pad_token_id,
    mask_token_id,
    batch_size,
    min_sen_len,
):
    """Get dataset after mlm from TFRecords"""

    def filter_by_length(x, min_sen_len):
        """Filter by minimum sentence length (subwords)"""
        return tf.squeeze(tf.greater_equal(tf.shape(x['input_ids']), tf.constant(min_sen_len)), axis=0)

    def filter_by_batch(x, y, batch_size):
        """Filter by batch size"""
        x_batch = tf.shape(x['input_ids'])[0]
        return tf.equal(x_batch, tf.constant(batch_size))

    dynamic_mlm_fn = dynamic_masking_from_features(
        max_seq_len,
        max_predictions_per_batch,
        vocab_size,
        cls_token_id,
        sep_token_id,
        unk_token_id,
        pad_token_id,
        mask_token_id,
    )

    train_dataset = get_tfdataset_from_tfrecords(tfrecord_path_list)
    if min_sen_len and min_sen_len > 0:
        train_dataset = train_dataset.filter(lambda x: filter_by_batch(x, min_sen_len))
    train_dataset = train_dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    train_dataset = train_dataset.map(dynamic_mlm_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.filter(lambda x, y: filter_by_batch(x, y, batch_size))
    train_dataset = train_dataset.shuffle(100)
    train_dataset = train_dataset.prefetch(100)

    return train_dataset


def get_model(vocab_size):
    """Model"""

    def model_fn():
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
            "type_vocab_size": 1,
            "vocab_size": vocab_size,
            "layer_norm_epsilon": 1e-12,
        }

        from tf_transformers.models import BertModel

        model = BertModel.from_config(config, use_masked_lm_positions=True, return_all_layer_outputs=True)

        print("Model inputs", model.input)
        print("Model outputs", model.output)
        return model

    return model_fn


def get_optimizer(learning_rate, train_steps, warmup_steps, optimizer_type):
    def optimizer_fn():
        optimizer, learning_rate_fn = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=train_steps,
            num_warmup_steps=warmup_steps,
            optimizer_type=optimizer_type,
        )

        return optimizer

    return optimizer_fn


def get_loss(loss_type):

    if loss_type == 'joint':

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

    else:

        def lm_loss(y_true_dict, y_pred_dict):
            """Joint loss over all layers"""
            loss_dict = {}
            loss = cross_entropy_loss(
                labels=y_true_dict['masked_lm_labels'],
                logits=y_pred_dict['token_logits'],
                label_weights=y_true_dict['masked_lm_weights'],
            )
            loss_dict['loss'] = loss
            return loss_dict

    return lm_loss


def get_trainer(device_type, device_address, dtype):

    if device_type == 'tpu':
        trainer = TPUTrainer(tpu_address=device_address, dtype=dtype)
        return trainer
    if device_type == 'gpu':
        pass


def train(cfg):

    # Load tokenizer from tf text SentencePieceTokenizer
    tokenizer_sp = load_tokenizer(cfg)

    # Vocab and tokens
    model_file_path = cfg.tokenizer.model_file_path
    vocab_size = cfg.tokenizer.vocab_size
    cls_id = tokenizer_sp._vocab[cfg.tokenizer.cls_token]
    mask_id = tokenizer_sp._vocab[cfg.tokenizer.mask_token]
    sep_id = tokenizer_sp._vocab[cfg.tokenizer.sep_token]
    unk_id = tokenizer_sp._vocab[cfg.tokenizer.unk_token]
    pad_id = tokenizer_sp._vocab[cfg.tokenizer.pad_token]

    # Data
    max_seq_len = cfg.data.max_seq_len
    max_predictions_per_batch = cfg.data.max_predictions_per_batch
    batch_size = cfg.data.batch_size
    min_sen_len = cfg.data.min_sen_len

    # Train Dataset
    tfrecord_path_list = cfg.data.tfrecord_path_list
    train_dataset = get_dataset(
        tfrecord_path_list,
        max_seq_len,
        max_predictions_per_batch,
        vocab_size,
        cls_id,
        sep_id,
        unk_id,
        pad_id,
        mask_id,
        batch_size,
        min_sen_len,
    )

    # Get Model
    model_fn = get_model(vocab_size)

    # Get Optimizer
    optimizer_fn = get_optimizer(
        cfg.model.optimizer.learning_rate,
        cfg.model.optimizer.train_steps,
        cfg.model.optimizer.warmup_steps,
        cfg.model.optimizer.optimizer_type,
    )

    # Get loss
    loss_fn = get_loss(cfg.model.loss.loss_type)
    training_loss_names = None
    if cfg.model.loss.loss_type == 'joint':
        training_loss_names = ['loss_{}'.format(i + 1) for i in range(12)]  # 12 num of hidden layers

    # Model params
    epochs = cfg.model.epochs
    steps_per_epoch = cfg.model.steps_per_epoch
    model_save_dir = cfg.model.model_save_dir
    callback_steps = cfg.model.callback_steps

    # Set callback
    # To use new sentencepiece model in T5 use like this
    t5_kwargs = {
        'bos_token': '[CLS]',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '[MASK]',
        'vocab_file': '{}'.format(model_file_path),
    }
    tokenizer_hf = T5Tokenizer(**t5_kwargs)
    tokenizer_hf.unique_no_split_tokens = tokenizer_hf.all_special_tokens
    mlm_callback = MLMCallback(tokenizer_hf)

    # Get trainer
    trainer = get_trainer(cfg.trainer.device_type, cfg.trainer.device_address, cfg.trainer.dtype)

    trainer.run(
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=train_dataset,
        train_loss_fn=loss_fn,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        model_checkpoint_dir=model_save_dir,  # gs://tft_free/
        batch_size=batch_size,
        training_loss_names=training_loss_names,
        validation_loss_names=None,
        validation_dataset=None,
        validation_loss_fn=None,
        validation_interval_steps=None,
        steps_per_call=100,
        enable_xla=False,
        callbacks=[mlm_callback],
        callbacks_interval_steps=callback_steps,
        overwrite_checkpoint_dir=True,
        max_number_of_models=10,
        model_save_interval_steps=None,
        repeat_dataset=True,
    )


@hydra.main(config_path="config", config_name="train_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    run()
