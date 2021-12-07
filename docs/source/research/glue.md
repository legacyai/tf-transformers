
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

# Glue Model Evaluation

GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
The leaderboard for the GLUE benchmark can be found [at this address](https://gluebenchmark.com/). It comprises the following tasks:

#### ax

A manually-curated evaluation dataset for fine-grained analysis of system performance on a broad range of linguistic phenomena. This dataset evaluates sentence understanding through Natural Language Inference (NLI) problems. Use a model trained on MulitNLI to produce predictions for this dataset.

#### cola

The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.

#### mnli

The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. The authors of the benchmark use the standard test set, for which they obtained private labels from the RTE authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. They also uses and recommend the SNLI corpus as 550k examples of auxiliary training data.

#### mnli_matched

The matched validation and test splits from MNLI. See the "mnli" BuilderConfig for additional information.

#### mnli_mismatched

The mismatched validation and test splits from MNLI. See the "mnli" BuilderConfig for additional information.

#### mrpc

The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

#### qnli

The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). The authors of the benchmark convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue.

#### qqp

The Quora Question Pairs2 dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent.

#### rte

The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges. The authors of the benchmark combined the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009). Examples are constructed based on news and Wikipedia text. The authors of the benchmark convert all datasets to a two-class split, where for three-class datasets they collapse neutral and contradiction into not entailment, for consistency.

#### sst2

The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.

#### stsb

The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data. Each pair is human-annotated with a similarity score from 1 to 5.

Note:
- [WNLI] (Winograd is not included in our evaluation code)


## How to run Evaluation using Tensorflow Transformers.

Evaluation GLUE using Tensorflow Transformers is fairly easy. The default code is written for ```AlbertModel```.
All the configuration are managed using [Hydra](https://github.com/facebookresearch/hydra).


- [GLUE Evaluation](https://github.com/legacyai/tf-transformers/tree/main/research/glue)


#### Evaluate using Joint Loss

```
python run_glue.py optimizer.loss_type=joint
```

#### Evaluate without Joint Loss

```
python run_glue.py
```

#### To run a task individually (eg: MRPC)
```
python run_glue.py +glue=mrpc glue.data.max_seq_length=128
```

After the code got executed there will be an output folder generated as per ```Hydra``` configuration.
Output folder looks like this ```outputs/2021-10-17/13-54-47/```

Upon succesful execution of code, output looks like this .


GLUE SCORE calculated
------------------------
|    |     cola |     mnli |     mrpc |   qnli |      qqp |   rte |   sst2 |     stsb |   glue_score |
|---:|---------:|---------:|---------:|-------:|---------:|------:|-------:|---------:|-------------:|
|  0 | 0.574227 | 0.8481   | 0.890    | 0.916  | 0.889    | 0.725 | 0.919  | 0.901    |   0.833384   |




GLUE SCORE calculated
------------------------
|          |      cola |     mnli |     mrpc |     qnli |      qqp |      rte |     sst2 |      stsb |   glue_score |
|:---------|----------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|-------------:|
| layer_1  | 0         | 0.581514 | 0.748025 | 0.612484 | 0.730661 | 0.552347 | 0.807339 | 0.0593384 |     0.511464 |
| layer_2  | 0.0181483 | 0.737464 | 0.777894 | 0.822259 | 0.834173 | 0.570397 | 0.869266 | 0.8295    |     0.682388 |
| layer_3  | 0.253889  | 0.78251  | 0.809552 | 0.859418 | 0.863381 | 0.588448 | 0.881881 | 0.862159  |     0.737655 |
| layer_4  | 0.378279  | 0.810607 | 0.845078 | 0.883397 | 0.874679 | 0.631769 | 0.905963 | 0.877547  |     0.775915 |
| layer_5  | 0.478266  | 0.82725  | 0.867434 | 0.896394 | 0.882526 | 0.642599 | 0.916284 | 0.889308  |     0.800008 |
| layer_6  | 0.518539  | 0.835905 | 0.879525 | 0.909024 | 0.886847 | 0.67509  | 0.918578 | 0.894333  |     0.81473  |
| layer_7  | 0.561713  | 0.842418 | 0.890978 | 0.913051 | 0.888815 | 0.700361 | 0.918578 | 0.898398  |     0.826789 |
| layer_8  | 0.573798  | 0.845268 | 0.892842 | 0.915431 | 0.889218 | 0.689531 | 0.917431 | 0.898625  |     0.827768 |
| layer_9  | 0.571642  | 0.846082 | 0.897044 | 0.914882 | 0.889671 | 0.700361 | 0.918578 | 0.901202  |     0.829933 |
| layer_10 | 0.566903  | 0.847608 | 0.895647 | 0.915614 | 0.889495 | 0.714801 | 0.919725 | 0.902506  |     0.831537 |
| layer_11 | 0.574227  | 0.848117 | 0.890978 | 0.916712 | 0.889708 | 0.725632 | 0.919725 | 0.901971  |     0.833384 |
| layer_12 | 0.573001  | 0.848117 | 0.887111 | 0.91598  | 0.889409 | 0.722022 | 0.920872 | 0.899414  |     0.831991 |


## How to change the base model?

Base model and tokenizer can be changed in ```model.py``` . But based on the special tokens, you might need to
modify some part of the code.
