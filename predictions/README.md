# Predictions
**Intermediate Outputs for Cascading**

## Purpose
Stores inference results (e.g., .pt or .npy files) to be used as inputs for subsequent stages (Cascading).

## Vertical Structure
1.  **Topic** (e.g., Source Counting)
2.  **Subtopic** (e.g., Journal 3)
3.  **Experiment** (The model that generated the predictions)
4.  **Dataset** (The data it was predicted on)

## Usage
*Example:* The **Source Counting** module writes estimated counts here. The **RTF Estimation** module reads them from here to use as input.
