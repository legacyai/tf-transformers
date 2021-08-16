

## Benchmark GPT2 with TFT and HuggingFace PT, TF and JAX


* Please make sure to install hydra-cc ,as all configs are managed using hydra
* By default all configurations are designed to run on GPU
* The following benchmarks ran on Volta V100 32 GB


### Run

a. **TFT** ( **bechmark.data.take_sample=true** , to run on  a small subset )

```
python run.py +benchmark=tft bechmark.data.take_sample=true
```

b. **HF PT**

```
python run.py +benchmark=hf bechmark.model.type=pt
```
c. **HF JAX**

```
python run.py +benchmark=hf bechmark.model.type=jax
```
d. **HF TF**

```
python run.py +benchmark=hf bechmark.model.type=tf
```




## Official Benchmarks on CNN DailyMail




### Greedy Decode
|                            |   batch_size | mode   |   max_length | time       |   samples/second |
|:---------------------------|-------------:|:-------|-------------:|:-----------|-----------------:|
| tf_transformers_serialized |           32 | greedy |           64 | 11 minutes |               17 |
| hf_tf                      |           32 | greedy |           64 | 37 minutes |                5 |
| hf_pt                      |           32 | greedy |           64 | 9 minutes  |               23 |
| hf_jax (model.generate)    |           32 | greedy |           64 | 27 minutes |                8 |
| hf_jax (pmap)              |           32 | greedy |           64 | 6 minutes  |               29 |




### Beam Decode


|                            |   batch_size | mode               |   max_length | time        | samples/second   |
|:---------------------------|-------------:|:-------------------|-------------:|:------------|:-----------------|
| tf_transformers_serialized |           32 | beam - num_beams=3 |           64 | 31 minutes  |                 8|
| hf_tf                      |           32 | beam - num_beams=3 |           64 | 83 minutes  |                 0|
| hf_pt                      |           32 | beam - num_beams=3 |           64 | 36 minutes  |                 7|
| hf_jax                     |           32 | beam - num_beams=3 |           64 | 35 minutes  |                 6|
