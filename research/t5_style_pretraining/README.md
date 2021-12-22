
### Sentence Masked Language Modeling using TFText

This folder has all necessary scripts to run the Sentence MLM using tensorflow text.
Instead of masking words, we mask sentences (sequence of words)

### Advantage

No need to prepare features in TFRecord format. We make use of dynamic mlm. All we need is a text file or folder of text files with text files, which each line corrsponding to text.

### WandB

By default we are using Wandb. if enviornment variable ```WANDB_PROJECT=None```, wandb will be disabled.

``` export WANDB_PROJECT='t5-style-pretraining' ```
### Configuration (Hydra)

All or most configurations can be managed using ```conf/config.yaml```. You can override it by command line also.

Eg: For TPU , we need a data in GCS and model_checkpoint_dir to be in GCS too.

``` nohup python3 run_t5_modified.py \
    task.data_directory=gs://legacyai-bucket \
    task.train_batch_size=128 \
    trainer.dtype=bf16 \
    trainer.model_checkpoint_dir=gs://legacyai-bucket/t5_style_t5_small \
    trainer.steps_per_epoch=10000 \
    trainer.epochs=100 \
    trainer.strategy=tpu \
    trainer.tpu_address=legacyai-tpu-1 \
    optimizer.learning_rate=0.01 \
    model.is_training=true \
    model.use_dropout=true \
    model.model_name=t5-small > logs &
    ```
