import logging
from pathlib import Path

import torch

from muse_toolbox.data.databases.ami_label_parsers import BaseAMILabelParser
from muse_toolbox.utils.audio_utils import load_audio

log = logging.getLogger(__name__)


class AMIDatabase:
    """Manages raw AMI audio files and parses ground truth labels."""

    def __init__(
        self,
        data_dir: str | Path,
        label_parser: BaseAMILabelParser,
        sampling_frequency: int,
    ) -> None:
        """Initializes the AMI database manager.

        Args:
            data_dir (str | Path): Path to the AMI database root.
            label_parser (BaseAMILabelParser): Instantiated label parser.
            sampling_frequency (int): Target sampling frequency in Hz.
        """
        self.data_dir = Path(data_dir)
        self.label_parser = label_parser
        self.fs = int(sampling_frequency)

    def get_meeting(
        self, meeting_id: str, array_id: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Loads multi-channel audio and sample-level SAD arrays for a meeting and array.

        Supports all AMI array topologies: 8-channel circular arrays (all rooms
        ``Array1`` and Edinburgh ``Array2``), 4-channel ceiling circular arrays
        (Idiap ``Array2``), and 10-channel linear arrays (TNO ``Array2``).

        Args:
            meeting_id (str): AMI meeting identifier (e.g., ``"EN2001a"``).
            array_id (str): Array identifier (e.g., ``"Array1"`` or ``"Array2"``).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - ``audio``: Audio waveform of shape ``(num_mics, num_samples)``.
                - ``sad_samples``: Dictionary mapping speaker IDs to boolean activity arrays.

        Raises:
            FileNotFoundError: If the first channel wav file is missing.
        """
        meeting_dir = self.data_dir / meeting_id / "audio"

        channels: list[torch.Tensor] = []
        # AMI arrays have up to 10 microphones (TNO Array2 has 10; Idiap Array2 has 4; others 8).
        for i in range(1, 11):
            wav_path = meeting_dir / f"{meeting_id}.{array_id}-{i:02d}.wav"
            if not wav_path.exists():
                if i > 1:
                    break
                raise FileNotFoundError(f"Missing audio file: {wav_path}")

            waveform, _ = load_audio(wav_path, sampling_frequency=self.fs)
            channels.append(waveform[0])

        audio = torch.stack(channels, dim=0)
        total_samples = int(audio.shape[1])
        sad_samples = self.label_parser.get_sad_samples(meeting_id, total_samples)

        return audio, sad_samples
