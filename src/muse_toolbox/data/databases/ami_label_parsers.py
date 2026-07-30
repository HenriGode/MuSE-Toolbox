import xml.etree.ElementTree as ET
from pathlib import Path
import torch
import logging

log = logging.getLogger(__name__)

class BaseAMILabelParser:
    """Base class for AMI label parsers using the Strategy Pattern."""
    
    def __init__(self, data_dir: str, sampling_frequency: int):
        self.data_dir = Path(data_dir)
        self.fs = sampling_frequency

    def get_sad_samples(self, meeting_id: str, total_samples: int) -> dict[str, torch.Tensor]:
        """
        Parses annotations and returns the sample-level SAD matrix.

        Args:
            meeting_id (str): The AMI meeting ID (e.g., 'EN2001a').
            total_samples (int): The total number of audio samples in this meeting.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping speaker IDs ('A', 'B', 'C', 'D', etc.) 
                                     to their binary activation arrays of shape (total_samples,).
        """
        raise NotImplementedError("Subclasses must implement get_sad_samples")


class AMISegmentsXMLParser(BaseAMILabelParser):
    """Parses the official segments.xml files for speaker activity."""
    
    def get_sad_samples(self, meeting_id: str, total_samples: int) -> dict[str, torch.Tensor]:
        # Path to segments directory
        segments_dir = self.data_dir / "Annotations" / "segments"
        
        sad_samples = {}
        
        # In AMI, speakers are typically A, B, C, D (sometimes E)
        for speaker_id in ["A", "B", "C", "D", "E"]:
            xml_file = segments_dir / f"{meeting_id}.{speaker_id}.segments.xml"
            if not xml_file.exists():
                continue
                
            # Create a zeroed boolean tensor for this speaker
            speaker_activity = torch.zeros(total_samples, dtype=torch.bool)
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # We can search for any element ending in 'segment' to ignore namespaces
            for segment in root.findall('.//*'):
                if segment.tag.endswith('segment') or segment.tag == 'segment':
                    if 'transcriber_start' in segment.attrib and 'transcriber_end' in segment.attrib:
                        start_s = float(segment.attrib['transcriber_start'])
                        end_s = float(segment.attrib['transcriber_end'])
                        
                        # Convert seconds to sample indices
                        start_idx = int(start_s * self.fs)
                        end_idx = int(end_s * self.fs)
                        
                        # Ensure indices are within bounds
                        start_idx = max(0, min(start_idx, total_samples - 1))
                        end_idx = max(0, min(end_idx, total_samples))
                        
                        if start_idx < end_idx:
                            speaker_activity[start_idx:end_idx] = True
            
            sad_samples[speaker_id] = speaker_activity
            
        if not sad_samples:
            log.warning(f"No segment annotations found for meeting {meeting_id}")
            
        return sad_samples
