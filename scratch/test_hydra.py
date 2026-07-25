import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Hydra logger name:", cfg.logger.name)

if __name__ == "__main__":
    main()
