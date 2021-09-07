from transformers import AlbertTokenizer

from tf_transformers.core import GPUTrainer, TPUTrainer
from tf_transformers.models import AlbertModel as Model
from tf_transformers.optimization import create_optimizer

MODEL_NAME = "albert-base-v2"


def get_model(return_all_layer_outputs, is_training, use_dropout):
    model = Model.from_pretrained(
        MODEL_NAME, return_all_layer_outputs=return_all_layer_outputs, is_training=is_training, use_dropout=use_dropout
    )
    return model


def get_tokenizer():
    return AlbertTokenizer.from_pretrained(MODEL_NAME)


def get_optimizer(learning_rate, examples, batch_size, epochs):
    steps_per_epoch = int(examples / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * num_train_steps)

    def optimizer_fn():
        optimizer, learning_rate_fn = create_optimizer(learning_rate, num_train_steps, num_train_steps)
        return optimizer

    return optimizer_fn


def get_trainer(device, dtype, strategy=None, num_gpus=None, tpu_address=None):

    if device not in ['gpu', 'tpu']:
        raise ValueError("Unknown device type {}".format(device))
    if device == 'tpu':
        if tpu_address is None:
            raise ValueError("When device is `tpu`, please provide tpu_address ('local' or ip address)")
        trainer = TPUTrainer(tpu_address=tpu_address, dtype=dtype)
        return trainer
    if device == 'gpu':
        if num_gpus is None:
            raise ValueError("When device is `gpu`, please provide num_gpus as int (1, 2 etc )")
        if strategy is None:
            strategy = 'mirrored'
        trainer = GPUTrainer(strategy, num_gpus=num_gpus, dtype=dtype)
        return trainer
