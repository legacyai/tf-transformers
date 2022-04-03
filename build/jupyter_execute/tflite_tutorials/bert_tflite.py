#!/usr/bin/env python
# coding: utf-8

# # Bert TFLite

# In[ ]:


get_ipython().system('pip install tf-transformers')

get_ipython().system('pip install sentencepiece')

get_ipython().system('pip install tensorflow-text')

get_ipython().system('pip install transformers')


# In[ ]:





# In[3]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
print("Tensorflow version", tf.__version__)

from tf_transformers.models import BertModel


# In[ ]:





# ## Convert a Model to TFlite
# 
# The most important thing to notice here is that, if we want to convert a model to ```tflite```, we have to ensure that ```inputs``` to the model are **deterministic**, which means inputs should not be dynamic. We have to fix  **batch_size**, **sequence_length** and other related input constraints depends on the model of interest.
# 
# ### Load Bert Model
# 
# 1. Fix the inputs
# 2. We can always check the ```model``` **inputs** and **output** by using ```model.input``` and ```model.output```.
# 3. We use ```batch_size=1``` and ```sequence_length=64```.)

# In[6]:


model_name = 'bert-base-cased'
batch_size = 1
sequence_length = 64
model = BertModel.from_pretrained(model_name, batch_size=batch_size, sequence_length=sequence_length)


# In[ ]:





# ## Verify Models inputs and outputs
# 

# In[7]:


print("Model inputs", model.input)
print("Model outputs", model.output)


# In[ ]:





# In[ ]:





# ## Save Model as Serialized Version
# 
# We have to save the model using ```model.save```. We use the ```SavedModel``` for converting it to ```tflite```.

# In[8]:


model.save("{}/saved_model".format(model_name), save_format='tf')


# In[15]:





# In[ ]:





# ## Convert SavedModel to TFlite

# In[9]:


converter = tf.lite.TFLiteConverter.from_saved_model("{}/saved_model".format(model_name)) # path to the SavedModel directory
converter.experimental_new_converter = True

tflite_model = converter.convert()

open("{}/saved_model.tflite".format(model_name), "wb").write(tflite_model)
print("TFlite conversion succesful")


# In[ ]:





# In[ ]:





# ## Load TFlite Model 

# In[10]:


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="{}/saved_model.tflite".format(model_name))
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[28]:





# In[ ]:





# ## Assert TFlite Model and Keras Model outputs
# 
# After conversion we have to assert the model outputs using
# ```tflite``` and ```Keras``` model, to ensure proper conversion.
# 
# 1. Create examples using ```tf.random.uniform```. 
# 2. Check outputs using both models.
# 3. Note: We need slightly higher ```rtol``` here to assert.

# In[15]:


# Dummy Examples 
input_ids = tf.random.uniform(minval=0, maxval=100, shape=(batch_size, sequence_length), dtype=tf.int32)
input_mask = tf.ones_like(input_ids)
input_type_ids = tf.zeros_like(input_ids)


# input type ids
interpreter.set_tensor(
    input_details[0]['index'],
    input_type_ids,
)
# input_mask
interpreter.set_tensor(input_details[1]['index'], input_mask)

# input ids
interpreter.set_tensor(
    input_details[2]['index'],
    input_ids,
)

# Invoke inputs
interpreter.invoke()
# Take last output
tflite_output = interpreter.get_tensor(output_details[-1]['index'])

# Keras Model outputs .
model_inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}
model_outputs = model(model_inputs)

# We need a slightly higher rtol here to assert :-)
tf.debugging.assert_near(tflite_output, model_outputs['token_embeddings'], rtol=3.0)
print("Outputs asserted and succesful:  ✅")


# In[ ]:




