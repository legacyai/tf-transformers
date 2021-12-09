

## Benchmark T5 with TFT and HuggingFace PT, TF and JAX


* Please make sure to install hydra-cc ,as all configs are managed using hydra
* By default all configurations are designed to run on GPU
* The following benchmarks ran on Volta V100 32 GB


### Run

a. **TFT** ( **bechmark.data.take_sample=true** , to run on  a small subset )

```
python run.py benchmark=tft benchmark.data.take_sample=true +benchmark.text_generation.eos_id=1
```

b. **HF PT**

```
python run.py +benchmark=hf benchmark.model.type=pt
```
For beam search

```
python run.py benchmark=hf benchmark.model.type=pt +benchmark.text_generation.num_beams=3

c. **HF JAX**

```
python run.py benchmark=hf benchmark.model.type=jax
```
d. **HF TF**

```
python run.py +benchmark=hf benchmark.model.type=tf
```




## Official Benchmarks on CNN DailyMail


{'model_type': 'textdecoder_serializable_tftext', 'time_taken': 517.600218296051, 'samples_per_second': 21.89720869382885}

### Greedy Decode
|                            |   batch_size | mode   |   max_length | time (sec) |   samples/second |
|:---------------------------|-------------:|:-------|-------------:|:-----------|-----------------:|
| tf_transformers_serialized |           32 | greedy |           64 | 389.81  |              29.47  |
| hf_tf                      |           32 | greedy |           64 | 2520    |               4     |
| hf_pt                      |           32 | greedy |           64 | 414.40  |              27.72  |
| hf_jax (model.generate)    |           32 | greedy |           64 | N/A     |                N/A  |
| hf_jax (pmap)              |           32 | greedy |           64 | N/A     |                N/A  |




### Beam Decode


|                            |   batch_size | mode               |   max_length | time (sec)  | samples/second   |
|:---------------------------|-------------:|:-------------------|-------------:|:------------|:-----------------|
| tf_transformers_serialized |           32 | beam - num_beams=3 |           64 | 740.89      |             15.50|
| hf_tf                      |           32 | beam - num_beams=3 |           64 | 5760        |                 2|
| hf_pt                      |           32 | beam - num_beams=3 |           64 | 1033.48     |             11.11|
| hf_jax                     |           32 | beam - num_beams=3 |           64 | N/A         |              N/A |
