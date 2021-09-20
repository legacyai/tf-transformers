import hydra
from mrpc import run_mrpc
from omegaconf import DictConfig


def run_glue(cfg):
    """All GLUE experiments starts here.
    We will train each task one by one.
    """

    # Run MRPC
    run_mrpc(cfg)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:

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
