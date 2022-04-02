---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Sentence Transformer in tf-transformers

* This is a simple tutorial to demonstrate how ```SentenceTransformer``` models has been integrated
to ```tf-transformers``` and how to use it
* The following tutorial is applicable to all supported ```SentenceTransformer``` models.

```{code-cell} ipython3

```

### Load Sentence-t5 model

```{code-cell} ipython3
import tensorflow as tf
from tf_transformers.models import SentenceTransformer
```

```{code-cell} ipython3
model_name = 'sentence-transformers/sentence-t5-base' # Load any sentencetransformer model here
model = SentenceTransformer.from_pretrained(model_name)
```

### Whats my model input?

* All models in ```tf-transformers``` are designed with full connections. All you need is ```model.input``` if its a ```LegacyModel/tf.keras.Model``` or ```model.model_inputs``` if its a ```LegacyLayer/tf.keras.layers.Layer```

```{code-cell} ipython3
model.input
```

### Whats my model output?

* All models in ```tf-transformers``` are designed with full connections. All you need is ```model.output``` if its a ```LegacyModel/tf.keras.Model``` or ```model.model_outputs``` if its a ```LegacyLayer/tf.keras.layers.Layer```

```{code-cell} ipython3
model.output
```

### Sentence vectors

```{code-cell} ipython3
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = ['This is a sentence to get vector', 'This one too']
inputs = tokenizer(text, return_tensors='tf', padding=True)

inputs_tf = {'input_ids': inputs['input_ids'], 'input_mask': inputs['attention_mask']}
outputs_tf = model(inputs_tf)
print("Sentence vector", outputs_tf['sentence_vector'].shape)
```

```{code-cell} ipython3

```

### Serialize as usual and load it

* Serialize, load and assert outputs with non serialized ```(```tf.keras.Model```)```

```{code-cell} ipython3
model_dir = 'MODELS/sentence_t5'
model.save_transformers_serialized(model_dir)

loaded = tf.saved_model.load(model_dir)
model = loaded.signatures['serving_default']

outputs_tf_serialized = model(**inputs_tf)

tf.debugging.assert_near(outputs_tf['sentence_vector'], outputs_tf_serialized['sentence_vector'])
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
