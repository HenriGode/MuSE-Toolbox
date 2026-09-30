import xml.etree.ElementTree as ET
from pathlib import Path
import torch
import logging

log = logging.getLogger(__name__)

class BaseAMILabelParser:
    """Base class for AMI label parsers using the Strategy Pattern."""
    
    def __init__(self, data_dir: str, sampling_frequency: int, annotations_dir_name: str = "Annotations_1.6.2_manual"):
        self.data_dir = Path(data_dir)
        self.fs = sampling_frequency
        self.annotations_dir_name = annotations_dir_name

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

    def _parse_xml_annotations(
        self,
        meeting_id: str,
        total_samples: int,
        sub_dir: str,
        file_suffix: str,
        target_tag: str,
        start_attr: str,
        end_attr: str
    ) -> dict[str, torch.Tensor]:
        """
        A shared helper method to parse AMI's standard XML annotation format.
        """
        xml_dir = self.data_dir / self.annotations_dir_name / sub_dir
        
        sad_samples = {}
        # In AMI, speakers are typically A, B, C, D (sometimes E)
        for speaker_id in ["A", "B", "C", "D", "E"]:
            xml_file = xml_dir / f"{meeting_id}.{speaker_id}.{file_suffix}"
            if not xml_file.exists():
                continue
                
            # Create a zeroed boolean tensor for this speaker
            speaker_activity = torch.zeros(total_samples, dtype=torch.bool)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # We can search for any element ending in target_tag to ignore namespaces
            for element in root.findall('.//*'):
                if element.tag.endswith(target_tag) or element.tag == target_tag:
                    if start_attr in element.attrib and end_attr in element.attrib:
                        start_s = float(element.attrib[start_attr])
                        end_s = float(element.attrib[end_attr])
                        
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
            log.warning(f"No {target_tag} annotations found for meeting {meeting_id}")
            
        return sad_samples


class AMISegmentsXMLParser(BaseAMILabelParser):
    """Parses the official segments.xml files for speaker activity."""
    
    def get_sad_samples(self, meeting_id: str, total_samples: int) -> dict[str, torch.Tensor]:
        return self._parse_xml_annotations(
            meeting_id=meeting_id,
            total_samples=total_samples,
            sub_dir="segments",
            file_suffix="segments.xml",
            target_tag="segment",
            start_attr="transcriber_start",
            end_attr="transcriber_end"
        )


class AMIWordsXMLParser(BaseAMILabelParser):
    """Parses the official words.xml files for high-precision word-level speech activity."""
    
    def get_sad_samples(self, meeting_id: str, total_samples: int) -> dict[str, torch.Tensor]:
        return self._parse_xml_annotations(
            meeting_id=meeting_id,
            total_samples=total_samples,
            sub_dir="words",
            file_suffix="words.xml",
            target_tag="w",
            start_attr="starttime",
            end_attr="endtime"
        )
