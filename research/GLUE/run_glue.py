import hydra
from hydra import compose
from mrpc import run_mrpc
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf")
def run_mrpc2(cfg):
    cfg2 = compose(overrides=["+glue=mrpc2"])
    print(cfg)
    print(OmegaConf.to_yaml(cfg2))


@hydra.main(config_path="conf")
def run_mrpc(cfg):
    cfg2 = compose(overrides=["+glue=mrpc"])
    print(cfg)
    print(OmegaConf.to_yaml(cfg2))


# def run_glue(cfg):
#     pass


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:

    run_mrpc(cfg)
    run_mrpc2(cfg)

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
