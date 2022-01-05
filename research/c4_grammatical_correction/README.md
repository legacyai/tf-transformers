
### Sentence Masked Language Modeling using TFText

This folder has all necessary scripts to run the Sentence MLM using tensorflow text.
Instead of masking words, we mask sentences (sequence of words)

### Advantage

No need to prepare features in TFRecord format. We make use of dynamic mlm. All we need is a text file or folder of text files with text files, which each line corrsponding to text.

### WandB

By default we are using Wandb. if enviornment variable ```WANDB_PROJECT=None```, wandb will be disabled.

``` export WANDB_PROJECT='t5-c4-grammatical-correction' ```
### Configuration (Hydra)

All or most configurations can be managed using ```conf/config.yaml```. You can override it by command line also.

Eg: For TPU , we need a data in GCS and model_checkpoint_dir to be in GCS too.

``` python3 run_c4_grammar_correction.py \
    task.data_directory=gs://legacyai-bucket/c4_grammar_correction_data \
    task.train_batch_size=512 \
    trainer.dtype=bf16 \
    trainer.model_checkpoint_dir=gs://legacyai-bucket/t5_c4_lr_3e5 \
    trainer.steps_per_epoch=50000 \
    trainer.epochs=10 \
    trainer.strategy=tpu \
    trainer.tpu_address=legacyai-tpu-2 \
    optimizer.learning_rate=3e-5 
    model.is_training=true
    model.use_dropout=true
    ```




