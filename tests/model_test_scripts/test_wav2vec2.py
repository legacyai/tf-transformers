
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
# import soundfile as sf
import torch

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")


from scipy.io import wavfile
import numpy as np

file_name = 'sample.wav'
data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/framerate
print('Sample rate:',framerate,'Hz')
print('Total time:',len(sounddata)/framerate,'s')

# Load file
import librosa
input_audio, _ = librosa.load(file_name, 
                              sr=16000)

input_values = processor(input_audio, return_tensors="pt").input_values # torch.Size([1, 3270299])

logits = model(input_values).logits # torch.Size([1, 10219, 32])

predicted_ids = torch.argmax(logits, dim=-1) # torch.Size([1, 10219])


transcription = processor.batch_decode(predicted_ids)[0]