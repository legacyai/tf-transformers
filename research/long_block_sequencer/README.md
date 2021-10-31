
### Long Block Sequencer using T5 and Bart for PubMed Summarization

This will take some time.

To prepare TFRecords for t5
```python3 prepare_tfrecords.py --model_name=t5-small --tfrecord_dir=t5_tfrecords```

To prepare TFRecords for bart
```python3 prepare_tfrecords.py --model_name=facebook/bart-base --tfrecord_dir=_bart_tfrecords```
