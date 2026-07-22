#!/bin/bash

# Navigate to the project root directory
cd /data4/Henri/MuSE-Toolbox

# Define the datasets to run
datasets=(
  "pra_anf_circ_8ch"
  "pra_anf_rand_8ch"
  "pra_anf_circ_2-8ch"
  "pra_anf_rand_2-8ch"
)

# Loop over the 4 datasets
for i in ${!datasets[@]}; do
    dataset=${datasets[$i]}
    gpu=$i  # Maps to GPU 0, 1, 2, and 3
    
    # We name the tmux session after the dataset
    session_name="J3_${dataset}"
    
    echo "🚀 Launching $dataset on GPU $gpu in tmux session: $session_name"
    
    # Create a new detached tmux session that runs the python command.
    # We use Hydra overrides to set the exact dataset and trainer.devices=[$gpu]
    tmux new-session -d -s $session_name "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate j3 && python scripts/main.py experiment=J3_pra_anf dataset=$dataset trainer.devices=[$gpu]"
    
    # Sleep briefly to ensure the session initializes without file I/O collisions
    sleep 5
done

echo ""
echo "✅ All 4 datasets are generating and training in separate tmux sessions!"
echo "----------------------------------------------------------------------"
echo "👉 To view a session, run: tmux attach -t J3_pra_anf_circ_8ch"
echo "👉 To detach from a session (leave it running in background), press: Ctrl+B, then D"
echo "👉 To see all active sessions, run: tmux ls"
