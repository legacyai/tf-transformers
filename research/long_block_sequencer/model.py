from long_block_encoder import Long_Block_Encoder

from tf_transformers.core import Trainer
from tf_transformers.losses.loss_wrapper import get_lm_loss
from tf_transformers.models import EncoderDecoder, T5Model, T5TokenizerTFText
from tf_transformers.optimization import create_optimizer


def get_model(model_name, num_splits, use_gru_layer, projection_dimension, return_all_layer_outputs):
    def model_fn():
        model = T5Model.from_pretrained(
            model_name, return_layer=True, decoder_kwargs={'return_all_layer_outputs': return_all_layer_outputs}
        )
        # Get encoder and decoder
        encoder = model._encoder
        decoder = model._decoder
        del model  # Free memory

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

    model = T5Model.from_pretrained(model_name, return_layer=True, use_auto_regressive=True)
    # Get encoder and decoder
    encoder = model._encoder
    decoder = model._decoder
    del model  # Free memory
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


def get_tokenizer(model_name, encoder_seq_length):
    tokenizer = T5TokenizerTFText.from_pretrained(
        model_name, max_length=encoder_seq_length, add_special_tokens=True, truncate=True
    )

    return tokenizer


def get_optimizer(
    learning_rate,
    steps_per_epoch,
    epochs,
    num_warmup_steps,
    decay_function='polynomial',
    weight_decay_rate=0.1,
    optimizer_type='adamw',
    use_constant_lr=False,
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
            weight_decay_rate=weight_decay_rate,
            end_learning_rate=end_learning_rate,
            optimizer_type=optimizer_type,
            use_constant_lr=use_constant_lr,
        )
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
