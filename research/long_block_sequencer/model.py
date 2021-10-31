from long_block_encoder import Long_Block_Encoder
from transformers import BartTokenizerFast, T5TokenizerFast

from tf_transformers.core import Trainer
from tf_transformers.losses.loss_wrapper import get_lm_loss
from tf_transformers.models import BartModel, EncoderDecoder, T5Model
from tf_transformers.optimization import create_optimizer


def get_model(model_name, num_splits, use_gru_layer, projection_dimension, return_all_layer_outputs):
    def model_fn():
        if model_name.startswith('bart') or model_name.startswith('facebook/bart'):
            model = BartModel.from_pretrained(
                model_name, return_layer=True, decoder_kwargs={'return_all_layer_outputs': return_all_layer_outputs}
            )
        elif model_name.startswith('t5'):
            model = T5Model.from_pretrained(
                model_name, return_layer=True, decoder_kwargs={'return_all_layer_outputs': return_all_layer_outputs}
            )
        else:
            raise ValueError("Unsupported model name {}".format(model_name))

        # Get encoder and decoder
        encoder = model._encoder
        decoder = model._decoder
        del model
        if use_gru_layer:
            long_model = Long_Block_Encoder(
                encoder, num_splits=num_splits, use_gru_layer=use_gru_layer, gru_units=projection_dimension
            )
        else:
            long_model = Long_Block_Encoder(
                encoder, num_splits=num_splits, use_gru_layer=use_gru_layer, dense_dimension=projection_dimension
            )

        decoder._embedding_layer = long_model.model_layer._embedding_layer
        model_encoder = EncoderDecoder(encoder=encoder, decoder=decoder)
        model_encoder = model_encoder.get_model()
        return model_encoder

    return model_fn


def get_model_inference(model_name, num_splits, use_gru_layer, projection_dimension):

    if model_name.startswith('bart') or model_name.startswith('facebook/bart'):
        model = BartModel.from_pretrained(model_name, return_layer=True, decoder_kwargs={'use_auto_regressive': True})
    elif model_name.startswith('t5'):
        model = T5Model.from_pretrained(model_name, return_layer=True, decoder_kwargs={'use_auto_regressive': True})
    else:
        raise ValueError("Unsupported model name {}".format(model_name))

    # Get encoder and decoder
    encoder = model._encoder
    decoder = model._decoder
    del model
    if use_gru_layer:
        long_model = Long_Block_Encoder(
            encoder, num_splits=num_splits, use_gru_layer=use_gru_layer, gru_units=projection_dimension
        )
    else:
        long_model = Long_Block_Encoder(
            encoder, num_splits=num_splits, use_gru_layer=use_gru_layer, dense_dimension=projection_dimension
        )

    decoder._embedding_layer = long_model.model_layer._embedding_layer
    model_encoder = EncoderDecoder(encoder=encoder, decoder=decoder)
    model_encoder = model_encoder.get_model()
    return model_encoder


def get_tokenizer(model_name):
    if model_name.startswith('bart') or model_name.startswith('facebook/bart'):
        tokenizer = BartTokenizerFast.from_pretrained(model_name)
    elif model_name.startswith('t5'):
        tokenizer = T5TokenizerFast.from_pretrained(model_name)
    else:
        raise ValueError("Unsupported model name {}".format(model_name))

    return tokenizer


def get_optimizer(learning_rate, examples, batch_size, epochs, use_constant_lr=False):
    """Get AdamW optimizer"""

    steps_per_epoch = int(examples / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * num_train_steps)

    def optimizer_fn():
        if use_constant_lr:
            from tf_transformers.optimization.adam_weighted import AdamWeightDecay

            optimizer = AdamWeightDecay(learning_rate=learning_rate)
            return optimizer

        optimizer, learning_rate_fn = create_optimizer(learning_rate, num_train_steps, warmup_steps)
        return optimizer

    return optimizer_fn


def get_loss(loss_type):
    """Get MLM Loss"""
    if loss_type and loss_type == "joint":
        return get_lm_loss(
            label_column='labels',
            label_weights_column='labels_mask',
            prediction_column='all_layer_token_logits',
            loss_type=loss_type,
        )
    return get_lm_loss(
        label_column='labels', label_weights_column='labels_mask', prediction_column='token_logits', loss_type=loss_type
    )


def get_trainer(distribution_strategy, dtype, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address, dtype=dtype)
    return trainer
