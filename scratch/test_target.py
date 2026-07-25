import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

def target_to_name(target_str):
    if not target_str:
        return "unknown"
    name = target_str.split('.')[-1].lower()
    name = name.replace("_feature_extractor", "")
    name = name.replace("channelcombinator", "")
    name = name.replace("estimator", "")
    if name.startswith("selfattention"):
        return "self_attention"
    return name

OmegaConf.register_new_resolver("tname", target_to_name, replace=True)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg) -> None:
    print("Run name:", cfg.logger.name)

if __name__ == "__main__":
    main()
