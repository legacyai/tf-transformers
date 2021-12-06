from tf_transformers.core import Trainer
from tf_transformers.losses.loss_wrapper import get_lm_loss_label_smoothing
from tf_transformers.models import T5Model, T5TokenizerTFText
from tf_transformers.optimization import create_optimizer

MODEL_NAME = 't5-small'


def get_model(return_all_layer_outputs, is_training, use_dropout, vocab_size, max_seq_len):
    """Get the model from model function"""

    def model_fn():
        model = T5Model.from_pretrained(MODEL_NAME)
        return model

    return model_fn


def get_tokenizer():
    """Get tokenizer"""
    tokenizer_layer = T5TokenizerTFText.from_pretrained(MODEL_NAME)
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
    loss_fn = get_lm_loss_label_smoothing(loss_type=loss_type)
    return loss_fn


def get_trainer(distribution_strategy, dtype, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address, dtype=dtype)
    return trainer
