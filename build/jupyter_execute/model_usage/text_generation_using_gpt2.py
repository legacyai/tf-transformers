#!/usr/bin/env python
# coding: utf-8

# # Text Generation using GPT2
# 
# * This tutorial is intended to provide, a familiarity in how to use ```GPT2``` for text-generation tasks.
# * No training is involved in this.

# In[ ]:


get_ipython().system('pip install tf-transformers')

get_ipython().system('pip install transformers')


# In[ ]:





# In[2]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
print("Tensorflow version", tf.__version__)
print("Devices", tf.config.list_physical_devices())

from tf_transformers.models import GPT2Model
from tf_transformers.text import TextDecoder
from transformers import GPT2Tokenizer


# In[ ]:





# ### Load GPT2 Model 
# 
# * 1. Note `use_auto_regressive=True`, argument. This is required for any models to enable text-generation.

# In[3]:


model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name, use_auto_regressive=True)


# In[ ]:





# ### Serialize and load
# 
# * The most recommended way of using a Tensorflow model is to load it after serializing.
# * The speedup, especially for text generation is up to 50x times.

# In[4]:


# Save as serialized
model_dir = 'MODELS/gpt2'
model.save_transformers_serialized(model_dir)

# Load
loaded = tf.saved_model.load(model_dir)
model = loaded.signatures['serving_default']


# In[ ]:





# ### Text-Generation
# 
# * . We can pass ```tf.keras.Model``` also to ```TextDecoder```, but this is recommended
# * . GPT2 like (Encoder) only models require ```-1``` as padding token.

# In[5]:


decoder = TextDecoder(model=loaded)


# ### Greedy Decoding

# In[34]:


texts = ['I would like to walk with my cat', 
         'Music has been very soothing']

input_ids = tf.ragged.constant(tokenizer(texts)['input_ids']).to_tensor(-1) # Padding GPT2 style models needs -1

inputs = {'input_ids': input_ids}
predictions = decoder.decode(inputs, 
                             mode='greedy', 
                             max_iterations=32)
print(tokenizer.batch_decode(tf.squeeze(predictions['predicted_ids'], axis=1)))


# ### Beam Decoding

# In[40]:


inputs = {'input_ids': input_ids}
predictions = decoder.decode(inputs, 
                             mode='beam',
                             num_beams=3,
                             max_iterations=32)
print(tokenizer.batch_decode(predictions['predicted_ids'][:, 0, :]))


# ### Top K Nucleus Sampling

# In[41]:


inputs = {'input_ids': input_ids}
predictions = decoder.decode(inputs, 
                             mode='top_k_top_p',
                             top_k=50,
                             top_p=0.7,
                             num_return_sequences=3,
                             max_iterations=32)
print(tokenizer.batch_decode(predictions['predicted_ids'][:, 0, :]))


# In[ ]:




