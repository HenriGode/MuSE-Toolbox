#!/usr/bin/env python3
"""Interactive diagnostic script comparing AMI manual XML annotation variants.

Structured with `#%%` cells so it can be run interactively in VS Code / Jupyter
or directly from the command line:

    python scripts/check_ami_annotations.py --no-plot
    python scripts/check_ami_annotations.py --plot
"""

#%% Imports, Logging & Official AMI Split Setup
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from muse_toolbox.data.databases.ami_label_parsers import (
    AMISegmentsXMLParser,
    AMIWordsXMLParser,
    BaseAMILabelParser,
)
from muse_toolbox.utils import STFTtransform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("check_ami_annotations")

CORNELL_TABLE1_PCT: dict[str, list[float]] = {
    "dev": [15.87, 67.17, 13.95, 2.59, 0.42],
    "eval": [15.12, 68.39, 12.63, 3.09, 0.76],
}


def load_ami_config(config_path: Path | None = None) -> dict[str, Any]:
    """Loads the AMI dataset configuration and resolves split meeting lists.

    Args:
        config_path (Path | None, optional): Optional path to ``ami.yaml``.
            Defaults to ``<PROJECT_ROOT>/configs/dataset/ami.yaml``.

    Returns:
        dict[str, Any]: Resolved AMI configuration dictionary.
    """
    cfg_path = config_path or (PROJECT_ROOT / "configs" / "dataset" / "ami.yaml")
    raw_cfg = OmegaConf.load(cfg_path)
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=False)
    assert isinstance(cfg_dict, dict)
    return {
        "ami_root": PROJECT_ROOT / "data" / "databases" / "ami",
        "val_meetings": list(cfg_dict["val_meetings"]),
        "test_meetings": list(cfg_dict["test_meetings"]),
        "fs": 16000,
    }


def get_meeting_num_samples(ami_root: Path, meeting_id: str) -> int:
    """Reads meeting sample duration quickly from the ``Array1-01.wav`` header.

    Args:
        ami_root (Path): Root directory of the AMI corpus.
        meeting_id (str): AMI meeting identifier.

    Returns:
        int: Total number of samples at native 16 kHz.
    """
    wav_path = ami_root / meeting_id / "audio" / f"{meeting_id}.Array1-01.wav"
    info = sf.info(str(wav_path))
    return int(round(info.frames * (16000.0 / float(info.samplerate))))


#%% Evaluate Annotation Variants vs. Cornell et al. (2022) Table 1
def compute_split_class_distribution(
    parser: BaseAMILabelParser,
    ami_root: Path,
    meetings: list[str],
    transform: STFTtransform,
    max_sources: int = 4,
) -> list[float]:
    """Computes frame-level class percentages (0..4 speakers) across a split.

    Args:
        parser (BaseAMILabelParser): Instantiated AMI XML label parser.
        ami_root (Path): Root directory of the AMI corpus.
        meetings (list[str]): Ordered list of meeting IDs in the split.
        transform (STFTtransform): STFT transform providing causal frame sampling.
        max_sources (int, optional): Upper clamp for source counts. Defaults to ``4``.

    Returns:
        list[float]: Class frequency percentages ``[0-spk, 1-spk, ..., K-spk]``.
    """
    counts = torch.zeros(max_sources + 1, dtype=torch.int64)
    for meeting_id in meetings:
        total_samples = get_meeting_num_samples(ami_root, meeting_id)
        sad_dict = parser.get_sad_samples(meeting_id, total_samples)
        if not sad_dict:
            continue
        sad_tensor = torch.stack(list(sad_dict.values()), dim=0)
        sad_frames = transform.samples2frames_quantity(sad_tensor, dim=1)
        frame_counts = torch.clamp(sad_frames.sum(dim=0), min=0, max=max_sources)
        counts += torch.bincount(frame_counts, minlength=max_sources + 1)

    total_frames = int(counts.sum().item())
    if total_frames == 0:
        return [0.0] * (max_sources + 1)
    return [100.0 * float(c.item()) / float(total_frames) for c in counts]


def run_annotation_comparison(ami_root: Path, val_meetings: list[str], test_meetings: list[str]) -> None:
    """Evaluates XML annotation variants on AMI dev/eval and logs Table 1 comparison.

    Args:
        ami_root (Path): Root directory of the AMI corpus.
        val_meetings (list[str]): Validation (``dev``) meeting IDs.
        test_meetings (list[str]): Evaluation (``eval``) meeting IDs.
    """
    transform = STFTtransform(
        frame_length=0.025,
        frame_shift=0.010,
        sampling_frequency=16000,
        window_type="hann",
        remove_DC=True,
        remove_Nyquist=True,
    )

    variants: list[tuple[str, BaseAMILabelParser]] = [
        (
            "segments.xml",
            AMISegmentsXMLParser(data_dir=ami_root, sampling_frequency=16000),
        ),
        (
            "words.xml (<w> only)",
            AMIWordsXMLParser(
                data_dir=ami_root,
                sampling_frequency=16000,
                include_vocalsounds=False,
                bridge_gaps_s=0.0,
            ),
        ),
        (
            "words.xml (<w> + <vocalsound>)",
            AMIWordsXMLParser(
                data_dir=ami_root,
                sampling_frequency=16000,
                include_vocalsounds=True,
                bridge_gaps_s=0.0,
            ),
        ),
        (
            "words.xml (<w>+<vocalsound>, gap<=0.10s)",
            AMIWordsXMLParser(
                data_dir=ami_root,
                sampling_frequency=16000,
                include_vocalsounds=True,
                bridge_gaps_s=0.10,
            ),
        ),
        (
            "words.xml (<w>+<vocalsound>, gap<=0.20s)",
            AMIWordsXMLParser(
                data_dir=ami_root,
                sampling_frequency=16000,
                include_vocalsounds=True,
                bridge_gaps_s=0.20,
            ),
        ),
    ]

    for split_name, meetings in (("dev", val_meetings), ("eval", test_meetings)):
        ref = CORNELL_TABLE1_PCT[split_name]
        log.info(f"\n=== AMI '{split_name}' Split ({len(meetings)} meetings) vs. Cornell Table 1 ===")
        header = (
            f"{'Annotation Variant':<42} | {'0-spk':>6} | {'1-spk':>6} | {'2-spk':>6} | "
            f"{'3-spk':>6} | {'4-spk':>6} | {'L1 vs Tab1':>10}"
        )
        log.info(header)
        log.info("-" * len(header))
        log.info(
            f"{'Cornell et al. (2022) Table 1':<42} | "
            f"{ref[0]:6.2f} | {ref[1]:6.2f} | {ref[2]:6.2f} | {ref[3]:6.2f} | {ref[4]:6.2f} | {'0.00':>10}"
        )
        for label, parser in variants:
            pcts = compute_split_class_distribution(parser, ami_root, meetings, transform)
            l1_diff = sum(abs(a - b) for a, b in zip(pcts, ref))
            log.info(
                f"{label:<42} | "
                f"{pcts[0]:6.2f} | {pcts[1]:6.2f} | {pcts[2]:6.2f} | "
                f"{pcts[3]:6.2f} | {pcts[4]:6.2f} | {l1_diff:10.2f}"
            )


#%% Optional Single-Meeting Headset Energy & XML Timeline Inspection
def inspect_headset_vs_xml(
    ami_root: Path,
    meeting_id: str = "EN2002a",
    duration_s: float = 60.0,
    save_path: Path | None = None,
) -> None:
    """Plots short-time Headset dB power against ``segments.xml`` and ``words.xml``.

    Args:
        ami_root (Path): Root directory of the AMI corpus.
        meeting_id (str, optional): Meeting ID to inspect. Defaults to ``"EN2002a"``.
        duration_s (float, optional): Initial segment length in seconds. Defaults to ``60.0``.
        save_path (Path | None, optional): Optional output image path.
    """
    import matplotlib.pyplot as plt

    fs = 16000
    num_samples = int(duration_s * fs)
    hop = 160
    num_frames = num_samples // hop

    seg_parser = AMISegmentsXMLParser(data_dir=ami_root, sampling_frequency=fs)
    word_parser = AMIWordsXMLParser(
        data_dir=ami_root,
        sampling_frequency=fs,
        include_vocalsounds=True,
        bridge_gaps_s=0.0,
    )
    seg_sad = seg_parser.get_sad_samples(meeting_id, num_samples)
    word_sad = word_parser.get_sad_samples(meeting_id, num_samples)

    speakers = sorted(seg_sad.keys())
    fig, axes = plt.subplots(len(speakers), 1, figsize=(14, 2.4 * len(speakers)), sharex=True)
    if len(speakers) == 1:
        axes = [axes]

    time_axis = np.arange(num_frames) * (hop / fs)
    for idx, (ax, spk) in enumerate(zip(axes, speakers)):
        headset_wav = ami_root / meeting_id / "audio" / f"{meeting_id}.Headset-{idx}.wav"
        if headset_wav.exists():
            wav, _ = sf.read(str(headset_wav), frames=num_samples)
            framed = wav[: num_frames * hop].reshape(num_frames, hop)
            power_db = 10.0 * np.log10(np.mean(framed**2, axis=1) + 1e-12)
            ax.plot(time_axis, power_db, linewidth=0.8, label=f"Headset-{idx} dB")

        seg_f = seg_sad[spk][: num_frames * hop : hop].numpy().astype(float)
        word_f = word_sad.get(spk, torch.zeros_like(seg_sad[spk]))[: num_frames * hop : hop].numpy().astype(float)
        ax.fill_between(time_axis, -80.0, -80.0 + 40.0 * seg_f, alpha=0.25, label="segments.xml")
        ax.fill_between(time_axis, -80.0, -80.0 + 30.0 * word_f, alpha=0.35, label="words.xml (<w>+<vocalsound>)")
        ax.set_ylabel(f"Spk {spk} (dB)")
        ax.set_ylim(-85.0, 0.0)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"AMI {meeting_id} Close-Talk Headset Energy vs. Manual XML Labels")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        log.info(f"Saved headset inspection plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    """Parses CLI flags for non-interactive terminal execution."""
    parser = argparse.ArgumentParser(
        description="Compare AMI XML annotation variants against Cornell et al. (2022) Table 1."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save the EN2002a headset vs. XML comparison plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (default when invoking from CLI without --plot).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ami_cfg = load_ami_config()
    ami_root: Path = ami_cfg["ami_root"]
    if not ami_root.exists():
        log.error(f"AMI database directory not found at {ami_root}")
        sys.exit(1)

    run_annotation_comparison(
        ami_root=ami_root,
        val_meetings=ami_cfg["val_meetings"],
        test_meetings=ami_cfg["test_meetings"],
    )
    if args.plot and not args.no_plot:
        inspect_headset_vs_xml(
            ami_root=ami_root,
            save_path=PROJECT_ROOT / "outputs" / "ami_annotation_inspection.png",
        )
