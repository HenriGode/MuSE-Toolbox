import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

def safe_get_choice(key: str, default: str = "unknown") -> str:
    try:
        return HydraConfig.get().runtime.choices.get(key, default)
    except Exception:
        return default

OmegaConf.register_new_resolver("get_choice", safe_get_choice, replace=True)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Resolved name:", cfg.logger.name)

if __name__ == "__main__":
    main()
