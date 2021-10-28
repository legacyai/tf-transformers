from tf_transformers.core import Trainer
from tf_transformers.losses.loss_wrapper import get_lm_loss
from tf_transformers.models import AlbertModel as Model
from tf_transformers.models import AlbertTokenizerTFText
from tf_transformers.optimization import create_optimizer

MODEL_NAME = "albert-large-v2"


def get_model(return_all_layer_outputs, is_training, use_dropout):
    """Get the model from model function"""

    def model_fn():
        model = Model.from_pretrained(
            MODEL_NAME,
            use_masked_lm_positions=True,
            return_all_layer_outputs=return_all_layer_outputs,
            is_training=is_training,
            use_dropout=use_dropout,
        )
        return model

    return model_fn


def get_tokenizer():
    tokenizer_layer = AlbertTokenizerTFText.from_pretrained(MODEL_NAME)
    return tokenizer_layer


def get_optimizer(learning_rate, examples, batch_size, epochs):
    """Get optimizer"""
    steps_per_epoch = int(examples / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * num_train_steps)

    def optimizer_fn():
        optimizer, learning_rate_fn = create_optimizer(learning_rate, num_train_steps, warmup_steps)
        return optimizer

    return optimizer_fn


def get_loss(loss_type):
    """Get MLM Loss"""
    return get_lm_loss(loss_type=loss_type)


def get_trainer(distribution_strategy, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address)
    return trainer
