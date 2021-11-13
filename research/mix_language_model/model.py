from mix_lm_model import MixEncoder

from tf_transformers.core import Trainer
from tf_transformers.losses.loss_wrapper import get_lm_loss
from tf_transformers.models import BigBirdRobertaTokenizerTFText, RobertaModel
from tf_transformers.optimization import create_optimizer

MODEL_NAME = 'roberta-base'
TOKENIZER_NAME = "google/bigbird-roberta-large"


def get_model(return_all_layer_outputs, is_training, use_dropout, vocab_size):
    """Get the model from model function"""

    def model_fn():
        # We use Roberta Style model, but we use BigBird Roberta Tokenizer
        config = RobertaModel.get_config(MODEL_NAME)
        # We update the vocab_size for that reason
        config['vocab_size'] = vocab_size
        config['type_vocab_size'] = -1  # We do not need type embeddings
        model = MixEncoder(
            config, return_all_layer_outputs=return_all_layer_outputs, is_training=is_training, use_dropout=use_dropout
        )
        model = model.get_model()
        return model

    return model_fn


def get_tokenizer():
    """Get tokenizer"""
    tokenizer_layer = BigBirdRobertaTokenizerTFText.from_pretrained(TOKENIZER_NAME)
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
    """Get MLM Loss"""
    return get_lm_loss(loss_type=loss_type)


def get_trainer(distribution_strategy, dtype, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address, dtype=dtype)
    return trainer
