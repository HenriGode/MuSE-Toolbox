import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


class BaseAMILabelParser:
    """Base class for AMI label parsers using the Strategy Pattern."""

    def __init__(
        self,
        data_dir: str | Path,
        sampling_frequency: int,
        annotations_dir_name: str = "Annotations_1.6.2_manual",
    ) -> None:
        """Initializes the base AMI XML label parser.

        Args:
            data_dir (str | Path): Root path to the AMI dataset directory.
            sampling_frequency (int): Target sample rate in Hz.
            annotations_dir_name (str, optional): Subdirectory containing manual
                XML annotations. Defaults to ``"Annotations_1.6.2_manual"``.
        """
        self.data_dir = Path(data_dir)
        self.fs = int(sampling_frequency)
        self.annotations_dir_name = annotations_dir_name

    def get_config(self) -> dict[str, Any]:
        """Returns a serializable configuration dictionary identifying the parser settings.

        Returns:
            dict[str, Any]: Parser class name and annotation directory parameters.
        """
        return {
            "parser": self.__class__.__name__,
            "annotations_dir_name": self.annotations_dir_name,
        }

    def get_sad_samples(
        self, meeting_id: str, total_samples: int
    ) -> dict[str, torch.Tensor]:
        """Parses annotations and returns the sample-level SAD matrix.

        Args:
            meeting_id (str): The AMI meeting ID (e.g., ``"EN2001a"``).
            total_samples (int): The total number of audio samples in this meeting.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping speaker IDs
                (``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"E"``) to their boolean
                activation arrays of shape ``(total_samples,)``.
        """
        raise NotImplementedError("Subclasses must implement get_sad_samples")

    @staticmethod
    def _element_matches_tags(tag: str, target_tags: tuple[str, ...]) -> bool:
        """Checks whether an XML element tag matches any requested tag name.

        Args:
            tag (str): XML element tag string (possibly namespace-prefixed).
            target_tags (tuple[str, ...]): Tag names to match.

        Returns:
            bool: ``True`` if the tag matches any target tag.
        """
        local_name = tag.rsplit("}", 1)[-1] if "}" in tag else tag
        return local_name in target_tags or tag in target_tags

    @staticmethod
    def _merge_and_bridge_intervals(
        intervals: list[tuple[int, int]], bridge_gap_samples: int
    ) -> list[tuple[int, int]]:
        """Merges overlapping intervals and bridges short silent gaps.

        Args:
            intervals (list[tuple[int, int]]): List of ``(start_idx, end_idx)``
                sample intervals.
            bridge_gap_samples (int): Maximum gap duration in samples to bridge
                between consecutive intervals.

        Returns:
            list[tuple[int, int]]: Sorted, merged list of sample intervals.
        """
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda pair: (pair[0], pair[1]))
        merged: list[tuple[int, int]] = [sorted_intervals[0]]
        for start_idx, end_idx in sorted_intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start_idx <= prev_end + bridge_gap_samples:
                merged[-1] = (prev_start, max(prev_end, end_idx))
            else:
                merged.append((start_idx, end_idx))
        return merged

    def _parse_xml_annotations(
        self,
        meeting_id: str,
        total_samples: int,
        sub_dir: str,
        file_suffix: str,
        target_tag: str | tuple[str, ...],
        start_attr: str,
        end_attr: str,
        bridge_gaps_s: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Parses AMI's standard XML annotations into per-speaker boolean tensors.

        Args:
            meeting_id (str): The AMI meeting ID (e.g., ``"EN2002a"``).
            total_samples (int): Total recording length in samples.
            sub_dir (str): Subdirectory inside ``annotations_dir_name``.
            file_suffix (str): XML filename suffix (e.g., ``"words.xml"``).
            target_tag (str | tuple[str, ...]): XML tag name(s) representing active
                intervals.
            start_attr (str): XML attribute holding interval start time in seconds.
            end_attr (str): XML attribute holding interval end time in seconds.
            bridge_gaps_s (float, optional): Maximum inactive gap in seconds to
                bridge between consecutive events for the same speaker. Defaults to ``0.0``.

        Returns:
            dict[str, torch.Tensor]: Per-speaker boolean activity vectors of shape
                ``(total_samples,)``.
        """
        xml_dir = self.data_dir / self.annotations_dir_name / sub_dir
        tags_tuple = (target_tag,) if isinstance(target_tag, str) else tuple(target_tag)
        bridge_gap_samples = max(0, int(round(bridge_gaps_s * self.fs)))

        sad_samples: dict[str, torch.Tensor] = {}
        # In AMI, speakers are typically A, B, C, D (sometimes E)
        for speaker_id in ["A", "B", "C", "D", "E"]:
            xml_file = xml_dir / f"{meeting_id}.{speaker_id}.{file_suffix}"
            if not xml_file.exists():
                continue

            speaker_activity = torch.zeros(total_samples, dtype=torch.bool)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            intervals: list[tuple[int, int]] = []
            for element in root.findall(".//*"):
                if self._element_matches_tags(element.tag, tags_tuple):
                    if start_attr in element.attrib and end_attr in element.attrib:
                        start_s = float(element.attrib[start_attr])
                        end_s = float(element.attrib[end_attr])

                        start_idx = int(start_s * self.fs)
                        end_idx = int(end_s * self.fs)

                        start_idx = max(0, min(start_idx, max(0, total_samples - 1)))
                        end_idx = max(0, min(end_idx, total_samples))

                        if start_idx < end_idx:
                            intervals.append((start_idx, end_idx))

            if bridge_gap_samples > 0 and intervals:
                intervals = self._merge_and_bridge_intervals(
                    intervals, bridge_gap_samples
                )

            for start_idx, end_idx in intervals:
                speaker_activity[start_idx:end_idx] = True

            sad_samples[speaker_id] = speaker_activity

        if not sad_samples:
            log.warning(f"No {tags_tuple} annotations found for meeting {meeting_id}")

        return sad_samples


class AMISegmentsXMLParser(BaseAMILabelParser):
    """Parses the official ``segments.xml`` files for speaker activity."""

    def get_sad_samples(
        self, meeting_id: str, total_samples: int
    ) -> dict[str, torch.Tensor]:
        """Parses segment-level transcriber annotations for a meeting.

        Args:
            meeting_id (str): The AMI meeting ID.
            total_samples (int): Total recording length in samples.

        Returns:
            dict[str, torch.Tensor]: Per-speaker boolean activity arrays.
        """
        return self._parse_xml_annotations(
            meeting_id=meeting_id,
            total_samples=total_samples,
            sub_dir="segments",
            file_suffix="segments.xml",
            target_tag="segment",
            start_attr="transcriber_start",
            end_attr="transcriber_end",
        )


class AMIWordsXMLParser(BaseAMILabelParser):
    """Parses the official ``words.xml`` files for word-level speech activity.

    Optionally includes ``<vocalsound>`` annotations (such as laughter or coughs)
    and bridges brief intra-phrase pauses up to ``bridge_gaps_s`` seconds.
    """

    def __init__(
        self,
        data_dir: str | Path,
        sampling_frequency: int,
        annotations_dir_name: str = "Annotations_1.6.2_manual",
        include_vocalsounds: bool = False,
        bridge_gaps_s: float = 0.0,
    ) -> None:
        """Initializes the word-level AMI XML label parser.

        Args:
            data_dir (str | Path): Root path to the AMI dataset directory.
            sampling_frequency (int): Target sampling frequency in Hz.
            annotations_dir_name (str, optional): Directory containing XML files.
                Defaults to ``"Annotations_1.6.2_manual"``.
            include_vocalsounds (bool, optional): If ``True``, also includes
                ``<vocalsound>`` XML tags that carry ``starttime`` and
                ``endtime`` attributes. Defaults to ``False``.
            bridge_gaps_s (float, optional): Maximum silent gap in seconds to
                bridge between consecutive annotations of the same speaker.
                Defaults to ``0.0``.
        """
        super().__init__(
            data_dir=data_dir,
            sampling_frequency=sampling_frequency,
            annotations_dir_name=annotations_dir_name,
        )
        if bridge_gaps_s < 0.0:
            raise ValueError(
                f"bridge_gaps_s must be non-negative, got {bridge_gaps_s}."
            )
        self.include_vocalsounds = bool(include_vocalsounds)
        self.bridge_gaps_s = float(bridge_gaps_s)

    def get_config(self) -> dict[str, Any]:
        """Returns a serializable configuration dictionary for word-level parser settings.

        Returns:
            dict[str, Any]: Parser class name, directory, vocal-sound, and gap-bridging settings.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "include_vocalsounds": self.include_vocalsounds,
                "bridge_gaps_s": self.bridge_gaps_s,
            }
        )
        return cfg

    def get_sad_samples(
        self, meeting_id: str, total_samples: int
    ) -> dict[str, torch.Tensor]:
        """Parses word-level (and optionally vocal-sound) annotations for a meeting.

        Args:
            meeting_id (str): The AMI meeting ID.
            total_samples (int): Total recording length in samples.

        Returns:
            dict[str, torch.Tensor]: Per-speaker boolean activity arrays.
        """
        target_tags: tuple[str, ...] = (
            ("w", "vocalsound") if self.include_vocalsounds else ("w",)
        )
        return self._parse_xml_annotations(
            meeting_id=meeting_id,
            total_samples=total_samples,
            sub_dir="words",
            file_suffix="words.xml",
            target_tag=target_tags,
            start_attr="starttime",
            end_attr="endtime",
            bridge_gaps_s=self.bridge_gaps_s,
        )
