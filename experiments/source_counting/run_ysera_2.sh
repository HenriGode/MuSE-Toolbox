#!/bin/bash
# Ysera Execution Script (by feature)
# Executes 4 datasets for each feature on 4 GPUs

SESSION="muse_ysera_sweep_features"
PROJECT_DIR="/data4/Henri/MuSE-Toolbox"

echo "Creating or attaching to tmux session: $SESSION"

# Create the session if it doesn't exist
tmux has-session -t $SESSION 2>/dev/null
if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION -n "init"
fi

## ========================================================
## Feature: pure_stft
## ========================================================
# tmux new-window -t $SESSION:1 -n "0_pure_stft_circ_8ch"
# tmux send-keys -t $SESSION:1 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[0] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:2 -n "1_pure_stft_circ_2-8ch"
# tmux send-keys -t $SESSION:2 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[1] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:3 -n "2_pure_stft_rand_8ch"
# tmux send-keys -t $SESSION:3 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[2] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:4 -n "3_pure_stft_rand_2-8ch"
# tmux send-keys -t $SESSION:4 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[3] model/feature_extractor=pure_stft model/channel_combinator=self_attention model/source_count_estimator=gru" C-m


## ========================================================
## Feature: log_mel
## ========================================================
# tmux new-window -t $SESSION:5 -n "0_log_mel_circ_8ch"
# tmux send-keys -t $SESSION:5 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[0] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:6 -n "1_log_mel_circ_2-8ch"
# tmux send-keys -t $SESSION:6 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[1] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:7 -n "2_log_mel_rand_8ch"
# tmux send-keys -t $SESSION:7 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[2] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:8 -n "3_log_mel_rand_2-8ch"
# tmux send-keys -t $SESSION:8 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[3] model/feature_extractor=log_mel model/channel_combinator=self_attention model/source_count_estimator=gru" C-m


## ========================================================
## Feature: ipd
## ========================================================
# tmux new-window -t $SESSION:9 -n "0_ipd_circ_8ch"
# tmux send-keys -t $SESSION:9 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[0] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:10 -n "1_ipd_circ_2-8ch"
# tmux send-keys -t $SESSION:10 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[1] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:11 -n "2_ipd_rand_8ch"
# tmux send-keys -t $SESSION:11 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[2] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:12 -n "3_ipd_rand_2-8ch"
# tmux send-keys -t $SESSION:12 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[3] model/feature_extractor=ipd model/channel_combinator=self_attention model/source_count_estimator=gru" C-m


## ========================================================
## Feature: gmsc
## ========================================================
# tmux new-window -t $SESSION:13 -n "0_gmsc_circ_8ch"
# tmux send-keys -t $SESSION:13 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[0] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:14 -n "1_gmsc_circ_2-8ch"
# tmux send-keys -t $SESSION:14 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[1] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:15 -n "2_gmsc_rand_8ch"
# tmux send-keys -t $SESSION:15 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[2] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

# tmux new-window -t $SESSION:16 -n "3_gmsc_rand_2-8ch"
# tmux send-keys -t $SESSION:16 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[3] model/feature_extractor=gmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m


## ========================================================
## Feature: wgmsc
## ========================================================
tmux new-window -t $SESSION:17 -n "1_wgmsc_circ_8ch"
tmux send-keys -t $SESSION:17 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_8ch trainer.devices=[1] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

tmux new-window -t $SESSION:18 -n "0_wgmsc_circ_2-8ch"
tmux send-keys -t $SESSION:18 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_circ_2-8ch trainer.devices=[0] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

tmux new-window -t $SESSION:19 -n "2_wgmsc_rand_8ch"
tmux send-keys -t $SESSION:19 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_8ch trainer.devices=[2] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m

tmux new-window -t $SESSION:20 -n "3_wgmsc_rand_2-8ch"
tmux send-keys -t $SESSION:20 "cd $PROJECT_DIR && DISPLAY=\"\" /home/henrig/.conda/envs/j3/bin/python scripts/main.py experiment=J3_pra_anf dataset=pra_anf_rand_2-8ch trainer.devices=[3] model/feature_extractor=wgmsc model/channel_combinator=self_attention model/source_count_estimator=gru" C-m


echo "Jobs have been sent to tmux session '$SESSION'."
echo "--------------------------------------------------------"
echo "To watch the runs live, attach to the session:"
echo "    tmux attach-session -t $SESSION"
echo ""
echo "Once attached, switch between windows using:"
echo "    Ctrl+B, then window number (1 to 20)"
echo "To detach and leave them running in the background:"
echo "    Ctrl+B, then d"
echo "--------------------------------------------------------"
