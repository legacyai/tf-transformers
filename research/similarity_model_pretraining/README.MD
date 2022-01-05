
### Masked Language Modeling using TFText

This folder has all necessary scripts to run the MLM using tensorflow text.

### Advantage

No need to prepare features in TFRecord format. We make use of dynamic mlm. All we need is a text file or folder of text files with text files, which each line corrsponding to text.

### Configuration (Hydra)

All or most configurations can be managed using ```conf/config.yaml```. You can override it by command line also.

Eg: For TPU , we need a data in GCS and model_checkpoint_dir to be in GCS too.

```python3 run_mlm.py \ data.data_directory=$GCP_BUCKET/data/ \ trainer.model_checkpoint_dir=$GCP_BUCKET/model```

### WandB

By default we are using Wandb. Check ```run_mlm.py``` to disable it.
