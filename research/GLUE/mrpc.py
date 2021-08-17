# mrpc
# The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically
# extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config/glue", config_name="mrpc.yaml")
def run_mrpc(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run_mrpc()
