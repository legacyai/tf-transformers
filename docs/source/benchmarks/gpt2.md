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

# Benchmark GPT2

- [Code - GPT2 Benchmark](https://github.com/legacyai/tf-transformers/tree/main/benchmark/gpt2)

This is used to benchmark the performance of GPT2 model on text generation tasks. We evaluate it using 3 frameworks.
Tensorflow-Transformers (default), HuggingFace PyTorch, HuggingFace Tensorflow and HuggingFace JAX.
Executing these scripts are fairly straightfoward and expect users to install the necessary libraries before executing
the benchmark script.

All the configuration are managed using [Hydra](https://github.com/facebookresearch/hydra).

[Machine] - Tesla V100-SXM2 32GB
[Tensorflow-version] - 2.4.1
[Huggingface-Transformer-Version] - 4.12.5
[PyTorch-Version] - 1.9.0

## Tensorflow-Transformers. (tft)

The default benchmark mode is ```tft```.
1. To execute ```Greedy Decoding``` (default) :
    ```python run.py benchmark=tft```

2. To execute ```Beam Decoding``` :
    ```python run.py benchmark=tft benchmark.text_generation.mode=beam +benchmark.text_generation.num_beams=3```

        * a. textdecoder_keras_model    -  Uses tf.keras.Model.
        * b. textdecoder_saved_model    -  Uses tf.saved_model, ```for``` loop to decode .
        * c. textdecoder_serializable   -  Uses tf.saved_model, ```model + text generation``` is serialized together.

3. To execute ```Greedy Decoding``` using let's say ```textdecoder_keras_model``` :
    ```python run.py benchmark=tft benchmark.model.type=textdecoder_keras_model```

[Note] - You can pass any arguments to ```tf_transformers.text.TextDecoder.decode``` arguments to the hydra configuration
using ```+``` .

## HuggingFace-Tensorflow. (hf-tf)

1. To execute ```Greedy Decoding``` (default) :
    ```python run.py benchmark=hf benchmark.model.type=tf```

2. To execute ```Beam Decoding``` :
    ```python run.py benchmark=hf benchmark.model.type=tf +benchmark.text_generation.num_beams=3 ```

[Note] - You can pass any arguments to ```model.generate``` arguments to the hydra configuration
using ```+``` .

## HuggingFace-PyTorch. (hf-pt)

1. To execute ```Greedy Decoding``` (default) :
    ```python run.py benchmark=hf benchmark.model.type=pt```

2. To execute ```Beam Decoding``` :
    ```python run.py benchmark=hf benchmark.model.type=pt +benchmark.text_generation.num_beams=3 ```


## HuggingFace-JAX. (hf-jax)

1. To execute ```Greedy Decoding``` (default) :
    ```python run.py benchmark=hf benchmark.model.type=jax```

2. To execute ```Beam Decoding``` :
    ```python run.py benchmark=hf benchmark.model.type=jax +benchmark.text_generation.num_beams=3 ```



## Official Benchmarks on CNN DailyMail

```
Greedy Decode:
|                            |   batch_size | mode   |   max_length | time       |   samples/second |
|:---------------------------|-------------:|:-------|-------------:|:-----------|-----------------:|
| tf_transformers_serialized |           32 | greedy |           64 | 11 minutes |               17 |
| tf_transformers + tf-text  |           32 | greedy |           64 | 12 minutes |               16 |
| hf_tf                      |           32 | greedy |           64 | 37 minutes |                5 |
| hf_pt                      |           32 | greedy |           64 | 9 minutes  |               23 |
| hf_jax (model.generate)    |           32 | greedy |           64 | 27 minutes |                8 |
| hf_jax (pmap)              |           32 | greedy |           64 | 6 minutes  |               29 |
```

```
Beam Decode:
|                            |   batch_size | mode               |   max_length | time        | samples/second   |
|:---------------------------|-------------:|:-------------------|-------------:|:------------|:-----------------|
| tf_transformers_serialized |           32 | beam - num_beams=3 |           64 | 31 minutes  |                 8|
| tf_transformers + tf-text  |           32 | greedy             |           64 | 31 minutes  |                 8|
| hf_tf                      |           32 | beam - num_beams=3 |           64 | 83 minutes  |                 0|
| hf_pt                      |           32 | beam - num_beams=3 |           64 | 36 minutes  |                 7|
| hf_jax                     |           32 | beam - num_beams=3 |           64 | 35 minutes  |                 6|
```
