
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

# Long Block Sequencer

Long Block Sequencer is a very simple yet an efficient way to make use of longer sequence in Attention based Transformer
models. The main reason, why Transformers are not able to scale to longer sequences is it because of the computational
cost while training. The ```softmax``` operation at every layer makes the gradient cacluation to cause Out of Memory
error.

Most recent works like BigBird, LongFormer etc has been developed on the top of concepts like Sparsity. And it does makes sense, because very long sequences have very less contextual connection. The sentence appear in the starting of long
sequence might have less connection with the last sequence. So, its a natural choice to have sparsity, moreover ```softmax``` on the top of these long sequence any way produce significantly smaller values.

The idea behind long block sequencer is simple. Instead of taking a long sequence as it is, try to split it internally
into equal chunks. Lets say ```sequence_length=4096``` and ```num_splits=8```, split these long sequence into ```8 (num_splits)``` sub sequence with ```512 sequence length (4097/8=512)``` each. Then process each sub sequence through the model of interest and concatanate all embeddings ```8 embeddings of batch_size x 512 x emb_dim ``` into a single embedding of size ``` batch_size x (4096) x emb_dim , where 4096 = 512 * 8 ``` matrix. Pass it through a feed forward network or RNN layer to have some non-linear interactions among them. This lets some information to pass through sub-embeddings during gradient calculation. This, final projection layer makes sure, all these indivudally processed sub embeddings get some interaction informaton while training. Then this final embedding can be used for any tasks.

This is explained through the following image:
![long-block-sequencer](../imgs/long_block_sequencer.gif)


### Advantages

1. This approach allows us to train models very easily and with no Out of Memory in most cases.
2. After training we can use same old architecture for inference for no or minimal changes.
3. Any existing pre-trained models can be used as base model for this approach, so its a very generelizable approach.
4 `t5-small` can be trained with ```sequence_length = 4096```, with batch_size of ```6``` in a ```Tesla V100 32 GB GPU```.

### Code and Results

- [T5 Long Block Sequencer](https://github.com/legacyai/tf-transformers/tree/main/research/long_block_sequncer)

We have trained Long Block Sequence with t5-small on PubMed Sumamrization. PubMed has very long Sentences, some even range
to sequence_length of 8000 subwords or more. The result show that, long range context is required for this task and we were
able to have ```Rogue-2 F1 Score``` better than Pegasus Base, BigBird-Roberta Base also. This seems to be a huge improvement, considering the size of the model ```t5-small```, which has ```60 million ``` parameters compared to Prgasus or BigBird-Roberta which has ```120 million``` parameters.

To train the model run
```python run_long_block_sequencer.py task.num_splits=8 trainer.model_checkpoint_dir=<YOUR_DIR>_splits-8 task.train_batch_size=8 ```

To evaluate the model for all checkpoints
```python evaluate.py eval.model_checkpoint_dir=<YOUR_DIR> ```

To evaluate only on a particular checkpoint
```python evaluate.py eval.model_checkpoint_path=<YOUR_CHECKPOINT_PATH> ```


[PubMed-Dataset](https://huggingface.co/datasets/pubmed)

Rogue SCORE
------------------------
|    |    Model         |seq_length|   Params |   R-2  |   R-1    |      R-l |
|---:|-----------------:|---------:|---------:|-------:|---------:|---------:|
|  0 |  t5-small        | 512      | 60M      | 0.07   |   N/A    |   N/A    |
|  0 |  LB-t5-small     | 4096     | 60M      | 17.41  |  41.89   |   26.44  |
|  0 | Pegasus-base     | 1024     | 120M     | 15.15  |  39.98   |   25.23  |
|  0 |BigBird-Roberta-B | 4096     | 120M     | 19.32  |  43.70   |   39.99  |

### Decoding and HyperParameter

We have used ```greedy decoding```. with ```decoder_sequence_length=256```. For training, we trained the model with constant ```learning_rate=0.001```.

### Reference
[1. BigBird](https://arxiv.org/pdf/2007.14062.pdf)


[2. Pegasus](https://arxiv.org/pdf/1912.08777.pdf)
