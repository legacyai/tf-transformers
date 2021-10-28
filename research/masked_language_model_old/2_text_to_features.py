import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from tf_transformers.data import TFWriter
from tf_transformers.text import SentencepieceTokenizer


def load_tokenizer(cfg):
    """Load tf text based tokenizer"""
    model_file_path = cfg.tokenizer.model_file_path
    do_lower_case = cfg.tokenizer.do_lower_case
    special_tokens = cfg.tokenizer.special_tokens

    tokenizer_layer = SentencepieceTokenizer(
        model_file_path=model_file_path, lower_case=do_lower_case, special_tokens=special_tokens
    )

    return tokenizer_layer


def create_tfrecords(cfg):
    """Prepare tfrecords"""
    schema = {
        "input_ids": ("var_len", "int"),
    }

    tfrecord_output_dir = cfg.data.tfrecord_output_dir
    tfrecord_filename = cfg.data.tfrecord_filename
    tfrecord_nfiles = cfg.data.tfrecord_nfiles
    tfrecord_mode = cfg.data.tfrecord_mode
    tfrecord_overwrite = cfg.data.tfrecord_overwrite

    input_text_files = cfg.data.input_text_files
    batch_size = cfg.data.batch_size

    tfwriter = TFWriter(
        schema=schema,
        file_name=tfrecord_filename,
        model_dir=tfrecord_output_dir,
        tag=tfrecord_mode,
        n_files=tfrecord_nfiles,
        overwrite=tfrecord_overwrite,
    )

    dataset = tf.data.TextLineDataset(input_text_files)

    def text_normalize(line):
        """Exclude empty string"""
        line = tf.strings.strip(line)
        return tf.not_equal(tf.strings.length(line), 0)

    dataset = dataset.filter(text_normalize)
    dataset = dataset.apply(tf.data.experimental.unique())
    dataset = dataset.batch(batch_size, drop_remainder=False)

    def parse_train():
        import tqdm

        tokenizer_layer = load_tokenizer(cfg)
        for batch_input in tqdm.tqdm(dataset):
            batch_input = {'text': [batch_input]}
            batch_tokenized = tokenizer_layer(batch_input)["input_ids"].to_list()
            for example_input_ids in batch_tokenized:
                yield {"input_ids": example_input_ids}

    # Process
    tfwriter.process(parse_fn=parse_train())


@hydra.main(config_path="config", config_name="tfrecord_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    create_tfrecords(cfg)


if __name__ == "__main__":
    run()
