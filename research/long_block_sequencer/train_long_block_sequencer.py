from dataset_loader import load_dataset, load_dataset_eval
from model import (
    get_loss,
    get_model,
    get_model_inference,
    get_optimizer,
    get_tokenizer,
    get_trainer,
)

from tf_transformers.callbacks.metrics import TextGenerationMetricCallback


def run_train(cfg, wandb):
    """Train function starts here

    Args:
        cfg (obj `DictConfig`): This is the config from hydra.
    """
    train_batch_size = cfg.data.train_batch_size
    dtype = cfg.trainer.dtype

    loss_type = cfg.optimizer.loss_type
    use_constant_lr = cfg.optimizer.use_constant_lr
    num_layers = cfg.model.num_layers
    return_all_layer_outputs = False
    training_loss_names = None
    if loss_type and loss_type == 'joint':
        return_all_layer_outputs = True
        training_loss_names = {'loss_{}'.format(i + 1) for i in range(num_layers)}

    learning_rate = cfg.optimizer.learning_rate
    epochs = cfg.trainer.epochs

    distribution_strategy = cfg.trainer.strategy
    num_gpus = cfg.trainer.num_gpus
    tpu_address = cfg.trainer.tpu_address
    model_checkpoint_dir = cfg.trainer.model_checkpoint_dir

    model_name = cfg.model.model_name
    num_splits = cfg.task.num_splits
    use_gru_layer = cfg.task.use_gru_layer
    projection_dimension = cfg.task.projection_dimension

    max_seq_len = cfg.task.max_seq_len
    decoder_seq_len = cfg.task.decoder_seq_len

    if max_seq_len % num_splits != 0:
        raise ValueError("`num_splits` should be divisble by `max_seq_len`")

    tokenizer_layer, tokenizer_hf = get_tokenizer(model_name, max_seq_len)
    train_dataset, total_examples = load_dataset(tokenizer_layer, max_seq_len, decoder_seq_len, train_batch_size)

    eval_dataset, _ = load_dataset_eval(tokenizer_layer, max_seq_len, decoder_seq_len, train_batch_size)
    eval_dataset = eval_dataset.take(20)  # We take only 20 after batching for callbacks

    # Get Model
    model_fn = get_model(model_name, num_splits, use_gru_layer, projection_dimension, return_all_layer_outputs)
    # Get Inference Model
    model_inference = get_model_inference(model_name, num_splits, use_gru_layer, projection_dimension)

    # Get Optimizer
    examples = total_examples  # Assume steps_per_epoch = 100000, and epochs = 5, examples = 500000
    optimizer_fn = get_optimizer(learning_rate, examples, train_batch_size, epochs, use_constant_lr)

    # Get loss
    loss_fn = get_loss(loss_type)

    # Get trainer
    trainer = get_trainer(
        distribution_strategy=distribution_strategy, num_gpus=num_gpus, tpu_address=tpu_address, dtype=dtype
    )

    # Define Callback
    # We use HF tokenizer to decode while generation
    callback = TextGenerationMetricCallback(model_inference, tokenizer_hf)

    steps_per_epoch = total_examples // train_batch_size
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
        repeat_dataset=True,
        wandb=wandb,
    )

    return history
