import hydra
from omegaconf import DictConfig
import os
import sys

# We will import the pipelines dynamically, but let's define the paths
# Note: You will need to implement the actual pipeline files in Step 2

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Standard Logging
    print(f"--> [MuSE-Toolbox] Starting Experiment: {cfg.get('experiment_name', 'Unnamed')}")
    print(f"--> [MuSE-Toolbox] Working Directory: {os.getcwd()}")
    
    # 2. Extract the Task ID from the config
    task = cfg.get("task", None)
    
    # 3. Dispatch to the specific Pipeline
    if task == "rtf_estimation":
        from muse_toolbox.pipelines import rtf_pipeline
        rtf_pipeline.run(cfg)
        
    elif task == "source_counting":
        from muse_toolbox.pipelines import counting_pipeline
        counting_pipeline.run(cfg)
        
    elif task == "beamforming":
        from muse_toolbox.pipelines import beamforming_pipeline
        beamforming_pipeline.run(cfg)
        
    else:
        print(f"❌ Error: Config missing valid 'task'. Got: {task}")
        sys.exit(1)

if __name__ == "__main__":
    main()