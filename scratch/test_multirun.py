import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

from omegaconf import OmegaConf
import hydra

def custom_get_choice(key: str, default: str = "unknown") -> str:
    prefix = f"{key}="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    
    from hydra.core.hydra_config import HydraConfig
    if HydraConfig.initialized():
        try:
            return HydraConfig.get().runtime.choices.get(key, default)
        except Exception:
            pass
    return default

OmegaConf.register_new_resolver("choice", custom_get_choice, replace=True)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg) -> None:
    print("Run name:", cfg.logger.name)

if __name__ == "__main__":
    main()
