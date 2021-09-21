import hydra
import pandas as pd
from mrpc import run_mrpc
from omegaconf import DictConfig


def flat_callbacks_to_df(history):
    callback_flatten = {}
    for item in history['callbacks']:
        for k, v in item[0].items():
            if k in callback_flatten:
                callback_flatten[k].append(v)
            else:
                callback_flatten[k] = [v]
    df = pd.DataFrame(callback_flatten)
    return df


def run_glue(cfg):
    """All GLUE experiments starts here.
    We will train each task one by one.
    """

    # Run MRPC
    history = run_mrpc(cfg)
    df_mrpcs = flat_callbacks_to_df(history)  # noqa

    return True


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # For TPU static padding is useful.
    if cfg.trainer.strategy == 'tpu':
        if cfg.data.static_padding is False:
            raise ValueError(
                "Make sure `static_padding` is set to True, when strategy is tpu. Please check config.yaml"
            )

    cfg_dict = dict(cfg)
    # If a specific task is specified, it must be under "glue" key
    # If not we will run the whole process
    if "glue" not in cfg_dict:
        run_glue(cfg)
    else:
        # Run mrpc
        if "mrpc" in cfg_dict["glue"]["task"]["name"]:
            run_mrpc(cfg)


if __name__ == "__main__":
    run()
