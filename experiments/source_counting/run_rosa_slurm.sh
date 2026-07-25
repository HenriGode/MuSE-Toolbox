#!/bin/bash
# Rosa Execution Script (SLURM HPC)
# Executes the remaining 15 combinations across the hfag.p partition using up to 8 GPUs concurrently

# IMPORTANT: Ensure that the hydra-submitit-launcher is installed in your conda environment on Rosa
# pip install hydra-submitit-launcher

echo "Submitting 15 runs to the Rosa SLURM queue..."

# We use Hydra multirun (-m) with the submitit_slurm launcher.
# SLURM will automatically manage scheduling the 15 jobs on the hfag.p partition.
# We set trainer.devices=[0] because SLURM will map the allocated GPU to CUDA_VISIBLE_DEVICES=0 for each job.

conda run -n j3 python scripts/main.py -m \
  experiment=J3_pra_anf \
  dataset=pra_anf_circ_2-8ch,pra_anf_rand_8ch,pra_anf_rand_2-8ch \
  model/feature_extractor=pure_stft,log_mel,ipd,gmsc,wgmsc \
  model/channel_combinator=self_attention \
  model/source_count_estimator=transformer \
  trainer.devices=[0] \
  hydra/launcher=submitit \
  hydra.launcher.partition=hfag.p \
  hydra.launcher.gres="gpu:L40S:1" \
  hydra.launcher.cpus_per_task=16 \
  hydra.launcher.mem_gb=64 \
  hydra.launcher.timeout_min=4320 \
  hydra.launcher.name="MuSE_Sweep" \
  hydra.launcher.array_parallelism=8

echo "Submission complete! You can monitor the jobs using 'squeue -u \$USER'"
