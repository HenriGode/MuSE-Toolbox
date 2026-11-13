# Real-Time Algorithmic Latency Design & Implementation Plan

## Philosophical Considerations

The transition from offline batch processing to real-time streaming requires a strict redefinition of algorithmic latency, particularly concerning how features (STFT) are extracted and how Ground Truth (GT) labels are generated.

### The `center=True` Dilemma
In offline processing, `torch.stft(..., center=True)` is standard. It aligns STFT frames perfectly with the original sample indices by padding the signal. 
- Frame 0 is centered exactly at Sample 0.
- Frame $f$ is centered at timestamp $t = f \times H$.

**The Real-Time Problem:** If frame $f$ mathematically requires audio samples up to $t + N/2$ (where $N$ is the FFT window size), we have inherently baked $N/2$ samples of **future knowledge (lookahead)** into the feature extraction. In a real-time system, you must physically wait $N/2$ samples past the "current time" to compute the frame, imposing a strict latency penalty.

### Causal Real-World Systems (`center=False`)
In a strict real-time causal system (e.g., WebRTC), we wait until the microphone buffer has $N$ samples, compute the FFT, and output the prediction. 
- The frame is not centered; it is "stamped" at the **end of the window**, as that is the earliest moment the system has all required information.
- By setting `center=False` in PyTorch's STFT, we can simulate this exact real-time behavior while maintaining offline batch-processing speedups.

### Ground Truth Alignment Drives Latency
Because the model predicts a single integer per frame (active speakers), latency depends on how the GT is generated from the SAD annotations:
- **Option A (Centered/Offline):** GT for frame $f$ is based on the center sample. The model learns to use future samples (the right-half of the window) to detect onsets early.
- **Option B (Causal/Real-Time):** GT for frame $f$ is based on the *end* of the window. The model learns to wait until the speaker is firmly in the buffer before changing its prediction.

### Algorithmic Latency vs. "Real" Detection Latency
1. **Algorithmic Latency (Lower Bound):** The absolute minimum physical delay dictated by the math (STFT windows, CNN lookahead). Evaluated statically (e.g., via Perturbation testing).
2. **Detection Latency (Practical Delay):** How many frames it actually takes for the model's internal activations to confidently detect an onset. Evaluated statistically via a Time-to-Detection (TTD) metric on the test set.

---

## Future Implementation Plan

When time permits, the following steps should be executed to migrate the framework to a mathematically rigorous, real-time-compatible architecture:

### 1. Refactor Feature Extraction and Ground Truth
- Modify `STFTtransform` to use `center=False` for strict causality without artificial lookahead.
- Refactor the Ground Truth generation logic to align labels based on the acoustic information available up to the *end* of the STFT window.

### 2. Rewrite Algorithmic Latency Callback
- Rename `CausalityCheckCallback` to `AlgorithmicLatencyCallback` (and flag to `check_algorithmic_latency`).
- Implement the **Perturbation Method** as a static unit test:
  - Generate a clean random signal and a perturbed clone (e.g., step function $+10.0$ at a random sample).
  - Extract STFT frames for both and find the first frame where they diverge ($F_{stft\_diff}$).
  - Process both through the model and find the first output frame where predictions diverge ($F_{out\_diff}$).
  - Compute Network Frame Latency: $L = F_{stft\_diff} - F_{out\_diff}$.
- Log the theoretical vs. empirical latency.

### 3. Implement Time-to-Detection (TTD) Metric
- Create a new evaluation metric (separate from the callback) that compares the timestamps of ground-truth speaker onsets against the model's predicted onsets during the test loop, calculating the average practical delay in milliseconds.
