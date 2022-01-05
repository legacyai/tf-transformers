
### Sentence Masked Language Modeling using TFText

This folder has all necessary scripts to run the Sentence MLM using tensorflow text.
Instead of masking words, we mask sentences (sequence of words)

### Advantage

No need to prepare features in TFRecord format. We make use of dynamic mlm. All we need is a text file or folder of text files with text files, which each line corrsponding to text.

### Configuration (Hydra)

All or most configurations can be managed using ```conf/config.yaml```. You can override it by command line also.

Eg: For TPU , we need a data in GCS and model_checkpoint_dir to be in GCS too.

```python3 run_mlm.py task.data_directory=<GCP_DATA_DIR> task.train_batch_size=128 trainer.dtype=bf16 trainer.model_checkpoint_dir=<GCP_MODEL_DIR> trainer.steps_per_epoch=50000 trainer.callback_steps=10000 trainer.epochs=20 trainer.strategy=tpu trainer.tpu_address=<TPU_NAME> optimizer.learning_rate=5e-4```

### WandB

By default we are using Wandb. if enviornment variable ```WANDB_PROJECT=None```, wandb will be disabled.
