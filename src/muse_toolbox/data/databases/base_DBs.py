"""Abstract database managers for MuSE-Toolbox.

Provides the foundational classes for source, noise, and RIR databases.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


class BaseDB(ABC):
    """The fundamental abstract base class for all database managers.

    It enforces that every database manager must have a root directory and
    a `prepare_data` method, which handles one-time setup like downloading
    or pre-processing.
    """

    def __init__(self, root_dir: str | Path) -> None:
        """Initializes the base database manager.

        Args:
            root_dir (Union[str, Path]): Path to the database root directory.
        """
        self.root_dir = Path(root_dir)

    @abstractmethod
    def prepare_data(self) -> None:
        """Performs all one-time setup for the database.
        
        This can include downloading, unzipping, pre-processing, or creating cache files.
        This method should be idempotent (safe to run multiple times).
        """
        pass


class BaseSourceDB(BaseDB, ABC):
    """Abstract base class for databases that provide clean source signals (e.g., speech)."""

    def __init__(
        self,
        root_dir: str | Path,
        signal_lengths: float,
    ) -> None:
        """Initializes the base source database manager.

        Args:
            root_dir (Union[str, Path]): Path to the database root directory.
            signal_lengths (float): Default length of signals to generate.
        """
        super().__init__(root_dir)
        self.signal_lengths = signal_lengths

    @abstractmethod
    def get_speaker_clips(self, split: str) -> dict[str, list[Any]]:
        """Returns a dictionary of speaker clips for a given split.

        Args:
            split (str): The dataset split (e.g., 'train', 'val', 'test').

        Returns:
            Dict[str, List[Any]]: A dictionary where keys are unique speaker IDs and
                values are lists of clips. A 'clip' can be a list of file paths or 
                any other identifier.
        """
        pass


class BaseRIRsDB(BaseDB, ABC):
    """Abstract base class for databases that provide Room Impulse Responses (RIRs)."""

    @property
    @abstractmethod
    def reverb_conditions(self) -> list[str] | list[float] | tuple[float, ...]:
        """Provides the available reverb conditions.

        Returns:
            Union[List[str], List[float], Tuple[float, ...]]: A list of all available 
                reverb condition identifiers (e.g., ['low', 'medium', 'high']).
        """
        pass

    @abstractmethod
    def get_rir(self, **kwargs: Any) -> torch.Tensor:
        """Returns the RIR tensor data based on given criteria.

        The specific keyword arguments depend on the concrete implementation
        (e.g., reverb_condition, doa, microphone_array).

        Args:
            **kwargs: Arbitrary keyword arguments filtering the RIR selection.

        Returns:
            torch.Tensor: The selected RIR tensor.
        """
        pass


class BaseNoiseDB(BaseDB, ABC):
    """Abstract base class for databases that provide noise signals."""

    @abstractmethod
    def get_noise(self, **kwargs: Any) -> torch.Tensor:
        """Returns the noise tensor data based on given criteria.

        The specific keyword arguments depend on the concrete implementation
        (e.g., reverb_condition, noise_type, microphone_array).

        Args:
            **kwargs: Arbitrary keyword arguments filtering the noise selection.

        Returns:
            torch.Tensor: The selected noise tensor.
        """
        pass
