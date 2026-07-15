import torch
from torchcodec.decoders import AudioDecoder
from pathlib import Path


def load_audio(
    filename: str | Path,
    signal_len: float | None = None,
    sampling_frequency: float = 16000.0,
) -> tuple[torch.Tensor, float]:
    """
    Loads an audio file, resamples, normalizes, and optionally truncates it.

    Args:
        filename (str): The path to the audio file.
        signal_len (float | None, optional): The desired length of the audio in seconds.
                                             If None, the full audio is loaded. Defaults to None.
        sampling_frequency (float, optional): The target sampling frequency. Defaults to 16000.0.

    Returns:
        tuple[torch.Tensor, float]: A tuple containing:
            - The processed audio data as a tensor (DC offset removed).
            - The sampling frequency of the audio data.
    """
    # Use torchcodec's AudioDecoder to load and resample the audio.
    decoder = AudioDecoder(source=filename, sample_rate=int(sampling_frequency))

    # Get samples up to signal_len if specified, otherwise get all samples.
    # The get_samples_played_in_range method handles truncation.
    samples = decoder.get_samples_played_in_range(stop_seconds=signal_len)
    audio_tensor = samples.data
    actual_sampling_rate = samples.sample_rate

    # Normalize the audio by removing the DC offset (subtracting the mean).
    # AudioDecoder normalizes to [-1, 1], but doesn't guarantee zero mean.
    mean_normalized_audio = audio_tensor - audio_tensor.mean(dim=-1, keepdim=True)

    return mean_normalized_audio, actual_sampling_rate
