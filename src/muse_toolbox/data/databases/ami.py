import torch
import torchaudio
import logging
from pathlib import Path
from muse_toolbox.data.databases.ami_label_parsers import BaseAMILabelParser
from muse_toolbox.utils.audio_utils import load_audio

log = logging.getLogger(__name__)

class AMIDatabase:
    """Manages raw AMI audio files and parses ground truth labels."""
    
    def __init__(self, data_dir: str, label_parser: BaseAMILabelParser, sampling_frequency: int):
        """
        Args:
            data_dir (str): Path to the AMI database root.
            label_parser (BaseAMILabelParser): Instantiated label parser.
            sampling_frequency (int): Target sampling frequency.
        """
        self.data_dir = Path(data_dir)
        self.label_parser = label_parser
        self.fs = sampling_frequency

    def get_meeting(self, meeting_id: str, array_id: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Loads the multi-channel audio and SAD matrix for a given meeting and array.
        
        Args:
            meeting_id (str): e.g., 'EN2001a'
            array_id (str): e.g., 'Array1' or 'Array2'
            
        Returns:
            tuple:
                audio (torch.Tensor): Audio waveform of shape [8, num_samples]
                sad_samples (dict[str, torch.Tensor]): Dict of speaker activity arrays.
        """
        meeting_dir = self.data_dir / meeting_id / "audio"
        
        channels = []
        # AMI arrays officially have 8 microphones, but Idiap Array2 only has 4.
        for i in range(1, 9):
            wav_path = meeting_dir / f"{meeting_id}.{array_id}-0{i}.wav"
            if not wav_path.exists():
                if i > 1:
                    break # We hit the end of the available microphones (e.g. 4 for Idiap)
                else:
                    raise FileNotFoundError(f"Missing audio file: {wav_path}")
                
            waveform, sample_rate = load_audio(wav_path, sampling_frequency=self.fs)
            channels.append(waveform[0]) # waveform is [1, num_samples], so take the 0th dim
            
        audio = torch.stack(channels, dim=0) # [8, num_samples]
        
        total_samples = audio.shape[1]
        
        # Get SAD samples using the injected strategy
        sad_samples = self.label_parser.get_sad_samples(meeting_id, total_samples)
        
        return audio, sad_samples
