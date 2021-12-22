import tensorflow as tf
from t5_modified import EncoderDecoderwithMLM
from t5_tokenizer_modified import T5CustomTokenizerTFText

from tf_transformers.core import Trainer
from tf_transformers.losses.loss_wrapper import get_lm_loss, get_lm_loss_label_smoothing
from tf_transformers.models import T5Encoder, T5Model
from tf_transformers.optimization import create_optimizer


def get_model(model_name, vocab_size, is_training, use_dropout):
    """Get the model from model function"""

    def model_fn():
        config = T5Model.get_config(model_name)
        encoder_config = config.copy()
        encoder_config['bidirectional'] = True
        encoder_config['vocab_size'] = vocab_size

        decoder_config = config.copy()
        decoder_config['bidirectional'] = False
        decoder_config['vocab_size'] = vocab_size

        encoder = T5Encoder(config=encoder_config, is_training=is_training, use_dropout=use_dropout)
        decoder = T5Encoder(
            config=decoder_config,
            use_decoder=use_dropout,
            mask_mode="causal",
            is_training=is_training,
            use_dropout=use_dropout,
        )

        # Share embeddings
        decoder._embedding_layer = encoder._embedding_layer

        model = EncoderDecoderwithMLM(
            encoder=encoder, decoder=decoder, cls_token_id=32000, is_training=is_training, use_dropout=use_dropout
        )

        model = model.get_model()

        return model

    return model_fn


def get_tokenizer(model_name):
    """Get tokenizer"""
    tokenizer_layer = T5CustomTokenizerTFText.from_pretrained(model_name)
    return tokenizer_layer


def get_optimizer(
    learning_rate,
    steps_per_epoch,
    epochs,
    num_warmup_steps,
    decay_function,
    adam_beta_1,
    adam_beta_2,
    adam_epsilon,
    weight_decay_rate,
    optimizer_type,
    use_constant_lr,
):
    """Get optimizer"""

    # Total train steps is steps_per_epoch * epochs
    num_train_steps = steps_per_epoch * epochs

    # Assuming warmup_steps is a ratio (float)
    if isinstance(num_warmup_steps, float):
        if num_warmup_steps < 1.0:
            num_warmup_steps = int(num_warmup_steps * num_train_steps)
        else:
            raise ValueError(
                "Provide num_warmup_steps is a float with value {}. Assuming\
                its a ratio , the value should be less than 1.0".format(
                    num_train_steps
                )
            )
    else:
        if isinstance(num_warmup_steps, int):
            pass
        else:
            raise TypeError("Unspported type {} for num_warmup_steps".format(type(num_warmup_steps)))

    # As in GPT2 paper, end_learning_rate = 0.1 * learning_rate
    end_learning_rate = 0.1 * learning_rate

    def optimizer_fn():
        optimizer, learning_rate_fn = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            decay_function=decay_function,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            adam_epsilon=adam_epsilon,
            weight_decay_rate=weight_decay_rate,
            end_learning_rate=end_learning_rate,
            optimizer_type=optimizer_type,
            use_constant_lr=use_constant_lr,
        )
        return optimizer

    return optimizer_fn


def get_loss(loss_type):
    """Get Language Model Loss"""

    lm_loss_fn = get_lm_loss_label_smoothing(
        label_column='labels', label_weights_column='labels_mask', prediction_column='decoder_token_logits'
    )
    mlm_loss_fn = get_lm_loss()

    def loss_fn_combined(batch_labels, model_outputs):

        # # cast logits loss to float32 for stability
        # encoder_logits_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=tf.range(tf.shape(model_outputs['logits'])[0]), logits=tf.cast(model_outputs['logits'], tf.float32)
        # )

        # # take transpose of logits
        # decoder_logits_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=tf.range(tf.shape(model_outputs['logits'])[0]),
        #     logits=tf.cast(tf.transpose(model_outputs['logits']), tf.float32),
        # )

        # logits_loss = (encoder_logits_loss + decoder_logits_loss) / 2.0
        print("Encoder token logits", model_outputs['encoder_token_logits'].shape)
        mlm_loss = mlm_loss_fn(batch_labels, {'token_logits': model_outputs['encoder_token_logits']})
        lm_loss = lm_loss_fn(batch_labels, model_outputs)

        loss_results = {}
        # loss_results['logits_loss'] = logits_loss
        loss_results['mlm_loss'] = mlm_loss['loss']
        loss_results['lm_loss'] = lm_loss['loss']
        # loss_results['loss'] = (loss_results['logits_loss'] + loss_results['mlm_loss'] + loss_results['lm_loss']) / 3.0
        loss_results['loss'] = (loss_results['mlm_loss'] + loss_results['lm_loss']) / 2.0

        return loss_results

    return loss_fn_combined



def get_trainer(distribution_strategy, dtype, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address, dtype=dtype)
    return trainer
