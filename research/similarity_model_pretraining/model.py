import tensorflow as tf
from similarity_model import Similarity_Model_Pretraining

from tf_transformers.core import Trainer
from tf_transformers.models import (
    BigBirdRobertaTokenizerTFText,
    GPT2Model,
    MaskedLMModel,
)
from tf_transformers.optimization import create_optimizer

MODEL_NAME = 'gpt2'
TOKENIZER_NAME = "google/bigbird-roberta-large"


def get_model(return_all_layer_outputs, is_training, use_dropout, vocab_size):
    """Get the model from model function"""

    def model_fn():
        # We use GPT2 Style model, but we use BigBird Roberta Tokenizer
        config = GPT2Model.get_config(MODEL_NAME)
        # We update the vocab_size for that reason
        config['vocab_size'] = vocab_size
        model = GPT2Model.from_config(
            config,
            mask_mode='user_defined',
            is_training=is_training,
            use_dropout=use_dropout,
            return_all_layer_outputs=return_all_layer_outputs,
            return_layer=True,
        )
        model = MaskedLMModel(
            model,
            use_extra_mlm_layer=False,
            hidden_size=config['embedding_size'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
        )
        model = Similarity_Model_Pretraining(encoder=model, projection_dimension=768)
        model = model.get_model()
        return model

    return model_fn


def get_tokenizer():
    tokenizer_layer = BigBirdRobertaTokenizerTFText.from_pretrained(TOKENIZER_NAME)
    return tokenizer_layer


def get_optimizer(learning_rate, steps_per_epoch, epochs, warmup_rate, learning_rate_type, use_constant_lr=False):
    """Get AdamW optimizer"""

    # Total steps over all epochs
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_rate * num_train_steps)
    if learning_rate_type is None:
        learning_rate_type = 'linear'

    def optimizer_fn():
        if use_constant_lr:
            from tf_transformers.optimization.adam_weighted import AdamWeightDecay

            optimizer = AdamWeightDecay(learning_rate=learning_rate)
            return optimizer

        optimizer, learning_rate_fn = create_optimizer(learning_rate, num_train_steps, warmup_steps, learning_rate_type)
        return optimizer

    return optimizer_fn


def get_loss(loss_type):
    """Get Similarity inbatch Loss"""

    def loss_fn(y_true_dict, y_pred_dict):
        """Loss function for in-batch loss"""

        batch_size = y_pred_dict['logits'].shape[0]  # Square matrix (batch_size x batch_size)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred_dict['logits'], labels=tf.range(batch_size))

        return {'loss': loss}

    return loss_fn


def get_trainer(distribution_strategy, dtype, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, dtype=dtype, num_gpus=num_gpus, tpu_address=tpu_address)
    return trainer
