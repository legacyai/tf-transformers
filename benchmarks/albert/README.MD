<!---
Copyright 2021 The TFT Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Benchmark Albert

- [Code - Albert Benchmark](https://github.com/legacyai/tf-transformers/tree/main/benchmark/albert)

This is used to benchmark the performance of Albert model on text generation tasks. We evaluate it using 3 frameworks.
Tensorflow-Transformers (default), HuggingFace PyTorch, HuggingFace Tensorflow and HuggingFace JAX (pending).
Executing these scripts are fairly straightfoward and expect users to install the necessary libraries before executing
the benchmark script.

All the configuration are managed using [Hydra](https://github.com/facebookresearch/hydra).

-> Machine - **Tesla V100-SXM2 32GB**

-> Tensorflow-version - **2.4.1**

-> Huggingface-Transformer-Version - **4.12.5**

-> PyTorch-Version - **1.9.0**

## Tensorflow-Transformers. (tft)

The default benchmark mode is ```tft```.
1. To execute ```tft``` (default) :
    ```python run.py benchmark=tft```

2. To execute ```type``` eg ```keras_model``` :
    ```python run.py benchmark=tft benchmark.model.type=keras_model```

        * a. keras_model    -  Uses tf.keras.Model.
        * b. saved_model    -  Uses tf.saved_model, ```for``` loop to decode .
        * c. saved_model_tftext   -  Uses tf.saved_model, ```model + text ``` is serialized together.


## HuggingFace-Tensorflow. (hf-tf)

1. To execute ```hf-tf``` (default) :
    ```python run.py benchmark=hf benchmark.model.type=tf```


## HuggingFace-PyTorch. (hf-pt)

1. To execute ```hf-pt``` (default) :
    ```python run.py benchmark=hf benchmark.model.type=pt```


## HuggingFace-JAX. (hf-jax) (Not Available)

1. To execute ```hf-jax``` (default) :
    ```python run.py benchmark=hf benchmark.model.type=jax```


## Official Benchmarks on IMDB

```
Greedy Decode:
|                            |   batch_size | mode   |   max_length | time (s)      |   samples/second |
|:---------------------------|-------------:|:-------|-------------:|:-----------|-----------------:|
| tf_transformers_serialized |           32 | greedy |           64 | 514 sec    |               22 |
| tf_transformers + tf-text  |           32 | greedy |           64 | 517 sec    |               22 |
| hf_tf                      |           32 | greedy |           64 | 1800 sec   |                6 |
| hf_pt                      |           32 | greedy |           64 | 278 sec    |               41 |
| hf_jax (model.generate)    |           32 | greedy |           64 | N/A        |              N/A |
| hf_jax (pmap)              |           32 | greedy |           64 | N/A        |              N/A |
```

```
Beam Decode:
|                            |   batch_size | mode               |   max_length | time        | samples/second   |
|:---------------------------|-------------:|:-------------------|-------------:|:------------|:-----------------|
| tf_transformers_serialized |           32 | beam - num_beams=3 |           64 | 503 sec     |                23|
| tf_transformers + tf-text  |           32 | beam - num_beams=3 |           64 | 509 sec     |                23|
| hf_tf                      |           32 | beam - num_beams=3 |           64 | 3240 sec    |                3 |
| hf_pt                      |           32 | beam - num_beams=3 |           64 | 660 sec     |                17|
```
