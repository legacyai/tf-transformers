from callbacks import MLMCallback
from dataset_loader import get_dataset
from model import (
    get_hf_tokenizer,
    get_loss,
    get_model,
    get_optimizer,
    get_tokenizer,
    get_trainer,
)

from tf_transformers.text.lm_tasks import mlm_fn


def run_train(cfg, wandb):
    """Train function starts here

    Args:
        cfg (obj `DictConfig`): This is the config from hydra.
    """

    data_directory = cfg.data.data_directory
    train_batch_size = cfg.data.train_batch_size
    max_seq_len = cfg.task.max_seq_len  # Maximum length per sequence
    max_predictions_per_seq = cfg.task.max_predictions_per_seq  # Maximum predictions (Mask) per sequence
    dtype = cfg.trainer.dtype

    is_training = cfg.model.is_training
    use_dropout = cfg.model.use_dropout
    loss_type = cfg.optimizer.loss_type
    use_constant_lr = cfg.optimizer.use_constant_lr
    num_layers = cfg.model.num_layers
    return_all_layer_outputs = False
    training_loss_names = None
    if loss_type and loss_type == 'joint':
        return_all_layer_outputs = True
        training_loss_names = {'loss_{}'.format(i + 1) for i in range(num_layers)}

    learning_rate = cfg.optimizer.learning_rate
    warmup_rate = cfg.optimizer.warmup_rate
    steps_per_epoch = cfg.trainer.steps_per_epoch
    epochs = cfg.trainer.epochs

    distribution_strategy = cfg.trainer.strategy
    num_gpus = cfg.trainer.num_gpus
    tpu_address = cfg.trainer.tpu_address
    model_checkpoint_dir = cfg.trainer.model_checkpoint_dir

    # Get dataset and tokenizer
    tokenizer_layer = get_tokenizer()
    # We split text by words (whitespace), inside MLM function.
    masked_lm_map_fn = mlm_fn(tokenizer_layer, max_seq_len, max_predictions_per_seq)
    train_dataset = get_dataset(data_directory, masked_lm_map_fn, train_batch_size)

    # Get Model
    model_fn = get_model(return_all_layer_outputs, is_training, use_dropout, tokenizer_layer.vocab_size.numpy())

    # Get Optimizer
    # steps_per_epoch is number of examples seen during one epoch (with batch size)
    # total examples per epoch = steps_per_epoch * batch_size
    examples = epochs * steps_per_epoch  # Assume steps_per_epoch = 100000, and epochs = 5, examples = 500000
    optimizer_fn = get_optimizer(learning_rate, examples, epochs, warmup_rate, use_constant_lr)
    # Get loss
    loss_fn = get_loss(loss_type)

    # Get trainer
    trainer = get_trainer(
        distribution_strategy=distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address, dtype=dtype
    )

    # Define Callback
    tokenizer = get_hf_tokenizer()
    callback = MLMCallback(tokenizer)
    callback_steps = 10000

    # Train
    history = trainer.run(
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=train_dataset,
        train_loss_fn=loss_fn,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        model_checkpoint_dir=model_checkpoint_dir,
        batch_size=train_batch_size,
        training_loss_names=training_loss_names,
        callbacks=[callback],
        callbacks_interval_steps=[callback_steps],
        repeat_dataset=True,
        wandb=wandb,
    )

    return history