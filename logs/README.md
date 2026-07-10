# Logs
**The Record of Execution**

## Purpose
Stores all automatic outputs generated during runtime for debugging.

## Vertical Structure
1.  **Topic** 
2.  **Subtopic** 
3.  **Experiment Name**
4.  **Run/Time**

## Contents
- **wandb/**: Weights & Biases data.
- **lightning_logs/**: PyTorch Lightning automated logs.
- **stdout/stderr**: Console outputs.

## Note
These files are ephemeral. Do not commit them to Git.
