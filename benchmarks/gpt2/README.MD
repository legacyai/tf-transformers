
1. Run benchmark_run.py

## GPT2 Benchmark


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
