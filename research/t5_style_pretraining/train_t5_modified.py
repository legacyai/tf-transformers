from dataset_loader import get_dataset
from model import get_loss, get_model, get_optimizer, get_tokenizer, get_trainer


def run_train(cfg, wandb):
    """Train function starts here

    Args:
        cfg (obj `DictConfig`): This is the config from hydra.
    """
    # Read Optimizer configs
    learning_rate = cfg.optimizer.learning_rate
    num_warmup_steps = cfg.optimizer.num_warmup_steps
    decay_function = cfg.optimizer.decay_function
    adam_beta_1 = cfg.optimizer.adam_beta_1
    adam_beta_2 = cfg.optimizer.adam_beta_2
    adam_epsilon = cfg.optimizer.adam_epsilon
    weight_decay_rate = cfg.optimizer.weight_decay_rate
    optimizer_type = cfg.optimizer.optimizer_type
    loss_type = cfg.optimizer.loss_type
    use_constant_lr = cfg.optimizer.use_constant_lr

    # Read trainer configs
    dtype = cfg.trainer.dtype
    num_gpus = cfg.trainer.num_gpus
    tpu_address = cfg.trainer.tpu_address
    epochs = cfg.trainer.epochs
    strategy = cfg.trainer.strategy
    steps_per_epoch = cfg.trainer.steps_per_epoch
    model_checkpoint_dir = cfg.trainer.model_checkpoint_dir
    global_norm = cfg.trainer.global_norm

    # Get model configs
    is_training = cfg.model.is_training
    model_name = cfg.model.model_name
    use_dropout = cfg.model.use_dropout

    # Get task specific configs
    # For mix LM we use max_seq_len as max_predictions_per_seq. Because we need
    # constant padding
    train_batch_size = cfg.task.train_batch_size
    data_directory = cfg.task.data_directory
    max_seq_len = cfg.task.max_seq_len

    # Get tokenizer
    tokenizer_layer = get_tokenizer(model_name)
    vocab_size = tokenizer_layer.vocab_size.numpy()

    # Get dataset
    train_dataset = get_dataset(data_directory, tokenizer_layer, max_seq_len, train_batch_size)

    # Get loss type
    training_loss_names = ['logits_loss', 'mlm_loss', 'lm_loss', 'loss']

    # Get model fn
    model_fn = get_model(model_name, vocab_size, is_training, use_dropout)

    # Get optimizer fn
    optimizer_fn = get_optimizer(
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
    )

    # Get loss fn
    loss_fn = get_loss(loss_type)

    # Get Trainer
    trainer = get_trainer(distribution_strategy=strategy, num_gpus=num_gpus, tpu_address=tpu_address, dtype=dtype)

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
        repeat_dataset=True,
        wandb=wandb,
        clip_norm=global_norm,
    )

    return history
