import logging
from pathlib import Path
from typing import Dict, List, Union, Any

from muse_toolbox.data.databases.base_DBs import BaseSourceDB

log = logging.getLogger(__name__)


class VCTKDatabase(BaseSourceDB):
    """Stub for the VCTK Database manager.
    
    Currently not fully implemented.
    """
    SPLIT_NAME_MAPPING = {
        "train": "train",
        "val": "val",
        "test": "test",
    }
    
    def __init__(self, root_dir: Union[str, Path], signal_lengths: float) -> None:
        super().__init__(root_dir, signal_lengths)

    def prepare_data(self) -> None:
        pass

    def get_speaker_clips(self, split: str) -> Dict[str, List[Any]]:
        return {}
