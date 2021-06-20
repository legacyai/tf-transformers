import logging

import hydra
from benchmark_tft import TftBenchmark
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)


def run_benchmark(cfg):

    if cfg.benchmark.task.name == "tft":

        benchmark = TftBenchmark(cfg)
        results = benchmark.run()
        log.info(results)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    run_benchmark((cfg))
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run()
