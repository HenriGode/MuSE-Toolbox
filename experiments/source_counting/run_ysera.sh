#!/bin/bash
# Ysera Execution Script (4 GPUs)
# Executes 5 combinations on the pra_anf_circ_8ch dataset using tmux

SESSION="muse_ysera_sweep"
PROJECT_DIR="/data4/Henri/MuSE-Toolbox"

# Create the session if it doesn't exist
tmux has-session -t $SESSION 2>/dev/null
if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION -n "init"
fi

# Create new detached session with the first window for GPU 0
# # GPU 0 takes pure_stft, and when finished, starts wgmsc
# # tmux send-keys -t $SESSION:0 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[0] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=transformer && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[0] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

## Dataset: pra_anf_circ_8ch

# # GPU 0 takes pure stft
# tmux new-session -d -s $SESSION -n "GPU_0"
# tmux send-keys -t $SESSION:0 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[0] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 1 takes log_mel
# tmux new-window -t $SESSION:1 -n "GPU_1"
# tmux send-keys -t $SESSION:1 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[1] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 2 takes ipd
# tmux new-window -t $SESSION:2 -n "GPU_2"
# tmux send-keys -t $SESSION:2 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[2] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 3 takes gmsc
# tmux new-window -t $SESSION:3 -n "GPU_3"
# tmux send-keys -t $SESSION:3 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[3] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 1 takes wgmsc
# tmux new-window -t $SESSION:4 -n "GPU_1"
# tmux send-keys -t $SESSION:4 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[1] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m


# ## Dataset = pra_anf_circ_2-8ch

# # GPU 0 takes pure stft
# tmux new-session -d -s $SESSION -n "GPU_0"
# tmux send-keys -t $SESSION:0 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[0] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 1 takes log_mel
# tmux new-window -t $SESSION:1 -n "GPU_1"
# tmux send-keys -t $SESSION:1 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[1] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 2 takes ipd
# tmux new-window -t $SESSION:5 -n "GPU_3"
# tmux send-keys -t $SESSION:5 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[3] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 3 takes gmsc
# tmux new-window -t $SESSION:3 -n "GPU_3"
# tmux send-keys -t $SESSION:3 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[3] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 1 takes wgmsc
# tmux new-window -t $SESSION:4 -n "GPU_1"
# tmux send-keys -t $SESSION:4 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[1] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m



# ## Dataset: pra_anf_rand_8ch

# # GPU 0 takes pure stft
# tmux new-session -d -s $SESSION -n "GPU_0"
# tmux send-keys -t $SESSION:0 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[0] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 1 takes log_mel
# tmux new-window -t $SESSION:1 -n "GPU_1"
# tmux send-keys -t $SESSION:1 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[1] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 2 takes ipd
# tmux new-window -t $SESSION:6 -n "GPU_0"
# tmux send-keys -t $SESSION:6 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[0] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 3 takes gmsc
# tmux new-window -t $SESSION:3 -n "GPU_3"
# tmux send-keys -t $SESSION:3 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[3] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 1 takes wgmsc
# tmux new-window -t $SESSION:4 -n "GPU_1"
# tmux send-keys -t $SESSION:4 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[1] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m


## Dataset = pra_anf_rand_2-8ch

# GPU 0 takes pure stft
tmux new-session -d -s $SESSION -n "GPU_0"
tmux send-keys -t $SESSION:0 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[0] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 1 takes log_mel
# tmux new-window -t $SESSION:7 -n "GPU_2"
# tmux send-keys -t $SESSION:7 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[2] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# # GPU 2 takes ipd
# tmux new-window -t $SESSION:2 -n "GPU_2"
# tmux send-keys -t $SESSION:2 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[2] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# GPU 3 takes gmsc
# tmux new-window -t $SESSION:3 -n "GPU_3"
# tmux send-keys -t $SESSION:3 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[3] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m

# GPU 1 takes wgmsc
tmux new-window -t $SESSION:4 -n "GPU_1"
tmux send-keys -t $SESSION:4 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[1] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=transformer" C-m






echo "All 5 runs launched in tmux session '$SESSION'."
echo "--------------------------------------------------------"
echo "To watch the runs live, attach to the session:"
echo "    tmux attach-session -t $SESSION"
echo ""
echo "Once attached, switch between GPUs using:"
echo "    Ctrl+B, then 0, 1, 2, or 3"
echo "To detach and leave them running in the background:"
echo "    Ctrl+B, then d"
echo "--------------------------------------------------------"
