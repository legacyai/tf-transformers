
### Long Block Sequencer using T5 and Bart for PubMed Summarization

To prepare TFRecords for t5
```python prepare_tfrecords.py model_name=t5 tfrecord_dir=t5_tfrecords```

To prepare TFRecords for bart
```python prepare_tfrecords.py model_name=facebook/bart-base tfrecord_dir=t5_tfrecords```
