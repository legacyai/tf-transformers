import tensorflow_text as tf_text  # noqa
import wandb
from dataset_loader import get_dataset
from model import get_model, get_optimizer, loss_fn

from tf_transformers.core import Trainer
from tf_transformers.models import AlbertTokenizerTFText

wandb.login()

TPU_ADDRESS = 'legacyai-tpu-2'
DTYPE = 'bf16'

delim_regex_pattern = '\. '  # noqa
window_length = 10
minimum_sentences = 4
batch_size = 512
max_seq_length = 256

learning_rate = 2e-5
epochs = 50
steps_per_epoch = 100000
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = 0.1 * num_train_steps
global_norm = 5.0
optimizer_fn = get_optimizer(learning_rate, num_train_steps, num_warmup_steps, decay_function='linear')

clip_logits = True
use_random_base = False

model_checkpoint_dir = 'gs://legacyai-bucket/sentence2vec_1'

WANDB_PROJECT = 'sentence2vec'
config_dict = {}
config_dict['learning_rate'] = learning_rate
config_dict['steps_per_epoch'] = steps_per_epoch
config_dict['epochs'] = epochs
config_dict['num_train_steps'] = steps_per_epoch * epochs
config_dict['num_warmup_steps'] = 0.1 * num_train_steps
config_dict['global_norm'] = global_norm
config_dict['model_checkpoint_dir'] = model_checkpoint_dir
config_dict['clip_logits'] = clip_logits
config_dict['use_random_base'] = use_random_base

wandb.init(project=WANDB_PROJECT, config=config_dict, sync_tensorboard=True)


trainer = Trainer(distribution_strategy='tpu', tpu_address=TPU_ADDRESS, dtype=DTYPE)

tokenizer_layer = AlbertTokenizerTFText.from_pretrained("albert-base-v2", add_special_tokens=False)
train_dataset = get_dataset(
    delim_regex_pattern, minimum_sentences, window_length, tokenizer_layer, max_seq_length, batch_size
)
model_fn = get_model(clip_logits, use_random_base)


# Train
training_loss_names = ['loss_cls', 'loss_mean', 'loss']
history = trainer.run(
    model_fn=model_fn,
    optimizer_fn=optimizer_fn,
    train_dataset=train_dataset,
    train_loss_fn=loss_fn,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    model_checkpoint_dir=model_checkpoint_dir,
    batch_size=batch_size,
    training_loss_names=training_loss_names,
    repeat_dataset=True,
    wandb=wandb,
    clip_norm=global_norm,
)
