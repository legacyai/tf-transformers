from tf_transformers.core import Trainer
from tf_transformers.losses.loss_wrapper import get_lm_loss
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
        model = GPT2Model.from_config(config, mask_mode='user_defined', return_layer=True)
        model = MaskedLMModel(
            model, hidden_size=config['embedding_size'], layer_norm_epsilon=config['layer_norm_epsilon']
        )
        model = model.get_model()
        return model

    return model_fn


def get_tokenizer():
    tokenizer_layer = BigBirdRobertaTokenizerTFText.from_pretrained(TOKENIZER_NAME)
    return tokenizer_layer


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
    return get_lm_loss(loss_type=loss_type)


def get_trainer(distribution_strategy, dtype, num_gpus=0, tpu_address=None):
    """Get Trainer"""
    trainer = Trainer(distribution_strategy, 
                      dtype=dtype,
                      num_gpus=num_gpus,
                      tpu_address=tpu_address)
    return trainer


def get_hf_tokenizer():
    """Get HuggingFace Tokenizer"""
    from transformers import BigBirdTokenizer

    tokenizer = BigBirdTokenizer.from_pretrained(TOKENIZER_NAME)
    return tokenizer
