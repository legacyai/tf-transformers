

```
python run_glue.py +glue=mrpc glue.data.max_seq_length=128
```

```
python run_glue.py optimizer.loss_type=joint
```

```
python run_glue.py optimizer.loss_type=joint data.take_sample=true
```

Output folder looks like this ```outputs/2021-10-17/13-54-47/```

