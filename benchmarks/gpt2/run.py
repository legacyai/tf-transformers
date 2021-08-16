import logging

import hydra
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)


def run_benchmark(cfg):

    if cfg.benchmark.task.name == "tft":
        from benchmark_tft import TftBenchmark

        benchmark = TftBenchmark(cfg)
        results = benchmark.run()
        log.info(results)

    if cfg.benchmark.task.name == "hf":
        from benchmark_hf import HFBenchmark

        benchmark = HFBenchmark(cfg)
        results = benchmark.run()
        log.info(results)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    run_benchmark((cfg))
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run()
