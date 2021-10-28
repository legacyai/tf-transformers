
### Steps

1. Prepare data . Using 2020, 2019 and 2018 data
    ```python prepare_data.py -s PROCESSED_AI_ML_FILES_2020/ -s 2020_mlm_sentences.txt```

2. ```export PYTHONPATH=PYTHONPATH$:~/Projects/tf-transformers/src/```


python run_mlm.py data.data_directory=data
