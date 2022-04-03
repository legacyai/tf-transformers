#!/usr/bin/env python
# coding: utf-8

# # Sentence Transformer in tf-transformers
# 
# * This is a simple tutorial to demonstrate how ```SentenceTransformer``` models has been integrated
# to ```tf-transformers``` and how to use it
# * The following tutorial is applicable to all supported ```SentenceTransformer``` models.

# In[ ]:





# ### Load Sentence-t5 model

# In[11]:


import tensorflow as tf
from tf_transformers.models import SentenceTransformer


# In[3]:


model_name = 'sentence-transformers/sentence-t5-base' # Load any sentencetransformer model here
model = SentenceTransformer.from_pretrained(model_name)


# ### Whats my model input?
# 
# * All models in ```tf-transformers``` are designed with full connections. All you need is ```model.input``` if its a ```LegacyModel/tf.keras.Model``` or ```model.model_inputs``` if its a ```LegacyLayer/tf.keras.layers.Layer```

# In[5]:


model.input


# ### Whats my model output?
# 
# * All models in ```tf-transformers``` are designed with full connections. All you need is ```model.output``` if its a ```LegacyModel/tf.keras.Model``` or ```model.model_outputs``` if its a ```LegacyLayer/tf.keras.layers.Layer```

# In[6]:


model.output


# ### Sentence vectors

# In[10]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = ['This is a sentence to get vector', 'This one too']
inputs = tokenizer(text, return_tensors='tf', padding=True)

inputs_tf = {'input_ids': inputs['input_ids'], 'input_mask': inputs['attention_mask']}
outputs_tf = model(inputs_tf)
print("Sentence vector", outputs_tf['sentence_vector'].shape)


# In[ ]:





# ### Serialize as usual and load it
# 
# * Serialize, load and assert outputs with non serialized ```(```tf.keras.Model```)```

# In[12]:


model_dir = 'MODELS/sentence_t5'
model.save_transformers_serialized(model_dir)

loaded = tf.saved_model.load(model_dir)
model = loaded.signatures['serving_default']

outputs_tf_serialized = model(**inputs_tf)

tf.debugging.assert_near(outputs_tf['sentence_vector'], outputs_tf_serialized['sentence_vector'])


# In[ ]:





# In[ ]:




