import hydra
from mrpc import run_mrpc
from omegaconf import DictConfig, OmegaConf


def run_glue(cfg):
    pass


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    pass
    # cfg_dict = dict(cfg)
    # # If a specific task is specified, it must be under "glue" key
    # # If not we will run the whole process
    # if "glue" not in cfg_dict:
    #     run_glue()
    # else:
    #     # Run mrpc
    #     if "mrpc" in cfg_dict["glue"]["task"]["name"]:
    #         run_mrpc(cfg)
    # print(OmegaConf.to_yaml(cfg))


# python run_glue.py +glue=mrpc glue.data.take_sample=true

if __name__ == "__main__":
    run()
