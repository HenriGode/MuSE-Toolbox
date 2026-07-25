from .timit import TIMITDatabase
from .vctk import VCTKDatabase
from .librispeech import LibrispeechDatabase
from .demand import DemandDatabase
from .local_noise import LocalNoiseDB
from .brudex_dbs import BrudexRIRsDB, BrudexNoiseDB

__all__ = [
    "TIMITDatabase",
    "VCTKDatabase",
    "LibrispeechDatabase",
    "DemandDatabase",
    "LocalNoiseDB",
    "BrudexRIRsDB",
    "BrudexNoiseDB",
]
