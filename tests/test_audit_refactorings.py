import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import yaml

from muse_toolbox.callbacks.save_results import SaveTestResultsCallback
from muse_toolbox.data.components.dynamic_chunk_dataset import DynamicChunkDataset
from muse_toolbox.data.databases.ami import AMIDatabase
from muse_toolbox.data.databases.ami_label_parsers import (
    AMISegmentsXMLParser,
    AMIWordsXMLParser,
)
from muse_toolbox.data.datamodules.ami import AMIDataModule
from muse_toolbox.metrics.source_counting.average_precision import AveragePrecision
from muse_toolbox.metrics.source_counting.eventF1 import TolerantEventF1
from muse_toolbox.models.components.feature_extractors import (
    CSIPD_Feature_Extractor,
    LogMel_Feature_Extractor,
)
from muse_toolbox.models.source_counting.estimators.transformer_estimator import (
    FusionTransformerEstimator,
    TransformerEstimator,
)
from muse_toolbox.models.source_counting.osc import OSCmodule
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


def _make_transform(fs: int = 16000) -> STFTtransform:
    """Builds a standard 16 kHz / 25 ms STFT transform matching Cornell baselines.

    Args:
        fs (int, optional): Sampling rate in Hz. Defaults to ``16000``.

    Returns:
        STFTtransform: Configured STFT transform instance.
    """
    return STFTtransform(
        frame_length=0.025,
        frame_shift=0.010,
        sampling_frequency=float(fs),
        window_type="torch.hann_window(x)",
        remove_DC=True,
        remove_Nyquist=True,
    )


def _write_synthetic_precomputed_split(
    split_dir: Path,
    recordings: dict[str, int],
    num_mics: int = 2,
    fs: int = 16000,
) -> None:
    """Writes synthetic precomputed numpy audio/SAD/metadata files for ``DynamicChunkDataset``.

    Args:
        split_dir (Path): Target split directory (e.g., ``tmp_path / "train"``).
        recordings (dict[str, int]): Mapping ``recording_key -> num_samples``.
        num_mics (int, optional): Number of microphone channels. Defaults to ``2``.
        fs (int, optional): Sampling frequency in Hz. Defaults to ``16000``.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx, (key, num_samples) in enumerate(recordings.items()):
        rng = np.random.default_rng(seed=100 + idx)
        audio = rng.normal(0.0, 0.05, size=(num_mics, num_samples)).astype(np.float32)
        sad = np.zeros((2, num_samples), dtype=bool)
        sad[0, : num_samples // 2] = True
        sad[1, num_samples // 4 :] = True
        np.save(split_dir / f"{key}_audio.npy", audio)
        np.save(split_dir / f"{key}_sad.npy", sad)
        meeting_id, array_id = key.split("_", 1) if "_" in key else (key, "Array1")
        torch.save(
            {
                "meeting_id": meeting_id,
                "array_id": array_id,
                "sampling_frequency": fs,
            },
            split_dir / f"{key}_meta.pt",
        )


def test_ami_database_10ch_array2_support(tmp_path: Path) -> None:
    """Verifies AMIDatabase loads up to 10 channels (D20 TNO Array2 linear array)."""
    fs = 16000
    num_samples = 3200
    meeting_id = "TS3003a"
    audio_dir = tmp_path / meeting_id / "audio"
    audio_dir.mkdir(parents=True)
    words_dir = tmp_path / "Annotations_1.6.2_manual" / "words"
    words_dir.mkdir(parents=True)

    for ch_idx in range(1, 11):
        wav_path = audio_dir / f"{meeting_id}.Array2-{ch_idx:02d}.wav"
        signal = np.full((num_samples,), fill_value=0.01 * ch_idx, dtype=np.float32)
        sf.write(str(wav_path), signal, fs)

    xml_path = words_dir / f"{meeting_id}.A.words.xml"
    xml_path.write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
        '<w nite:id="1" starttime="0.02" endtime="0.10">hello</w>'
        "</nite:root>",
        encoding="utf-8",
    )

    parser = AMIWordsXMLParser(data_dir=tmp_path, sampling_frequency=fs)
    db = AMIDatabase(data_dir=tmp_path, label_parser=parser, sampling_frequency=fs)
    audio, sad_samples = db.get_meeting(meeting_id, "Array2")

    assert audio.shape == (10, num_samples)
    assert "A" in sad_samples
    assert sad_samples["A"].shape == (num_samples,)
    assert int(sad_samples["A"].sum().item()) == int(0.08 * fs)


def test_ami_only_8ch_circular_filter(tmp_path: Path) -> None:
    """Verifies only_8ch_circular filters out non-8ch Array2 rooms (D20)."""
    fs = 16000
    transform = _make_transform(fs=fs)
    dummy_db = MagicMock()
    base_kwargs: dict[str, Any] = {
        "data_dir": str(tmp_path),
        "precomputed_dir": str(tmp_path),
        "transform": transform,
        "batch_size": 2,
        "num_workers": 0,
        "generation_config": {},
        "seed": 42,
        "reset": False,
        "acc_device": "cpu",
    }
    dm_filtered = AMIDataModule(
        database=dummy_db,
        train_meetings=["EN2001a", "ES2002a", "IS1008a", "TS3003a"],
        val_meetings=["ES2003a", "TS3004a"],
        test_meetings=["IB4001", "EN2002a"],
        chunk_length_s=2.0,
        min_context_s=1.0,
        train_items_per_epoch=None,
        arrays=["Array1", "Array2"],
        augmentation={"enabled": False},
        sampling_frequency=fs,
        only_8ch_circular=True,
        **base_kwargs,
    )

    assert dm_filtered._is_allowed_meeting_array("EN2001a", "Array1") is True
    assert dm_filtered._is_allowed_meeting_array("EN2001a", "Array2") is True
    assert dm_filtered._is_allowed_meeting_array("ES2002a", "Array2") is True
    assert dm_filtered._is_allowed_meeting_array("IS1008a", "Array1") is True
    assert dm_filtered._is_allowed_meeting_array("TS3003a", "Array1") is True
    assert dm_filtered._is_allowed_meeting_array("IS1008a", "Array2") is False
    assert dm_filtered._is_allowed_meeting_array("TS3003a", "Array2") is False
    assert dm_filtered._is_allowed_meeting_array("IB4001", "Array2") is False

    allowed_train = dm_filtered._get_allowed_keys_for_split("train")
    assert allowed_train == {
        "EN2001a_Array1",
        "EN2001a_Array2",
        "ES2002a_Array1",
        "ES2002a_Array2",
        "IS1008a_Array1",
        "TS3003a_Array1",
    }

    dm_unfiltered = AMIDataModule(
        database=dummy_db,
        train_meetings=["EN2001a", "IS1008a"],
        val_meetings=[],
        test_meetings=[],
        chunk_length_s=2.0,
        min_context_s=1.0,
        train_items_per_epoch=None,
        arrays=["Array1", "Array2"],
        augmentation={"enabled": False},
        sampling_frequency=fs,
        only_8ch_circular=False,
        **base_kwargs,
    )
    assert dm_unfiltered._is_allowed_meeting_array("IS1008a", "Array2") is True
    assert dm_unfiltered._is_allowed_meeting_array("TS3003a", "Array2") is True


def test_ami_words_xml_parser_vocalsounds_and_bridge_gaps(tmp_path: Path) -> None:
    """Verifies AMIWordsXMLParser include_vocalsounds and bridge_gaps_s options (D21)."""
    fs = 1000
    total_samples = 1000
    words_dir = tmp_path / "Annotations_1.6.2_manual" / "words"
    words_dir.mkdir(parents=True)
    xml_path = words_dir / "TEST01.A.words.xml"
    xml_path.write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
        '<w starttime="0.10" endtime="0.30">one</w>'
        '<vocalsound starttime="0.40" endtime="0.50" type="laugh"/>'
        '<w starttime="0.60" endtime="0.80">two</w>'
        "</nite:root>",
        encoding="utf-8",
    )

    parser_w_only = AMIWordsXMLParser(
        data_dir=tmp_path,
        sampling_frequency=fs,
        include_vocalsounds=False,
        bridge_gaps_s=0.0,
    )
    sad_w = parser_w_only.get_sad_samples("TEST01", total_samples)["A"]
    assert int(sad_w.sum().item()) == 400
    assert not bool(sad_w[450].item())

    parser_with_vocal = AMIWordsXMLParser(
        data_dir=tmp_path,
        sampling_frequency=fs,
        include_vocalsounds=True,
        bridge_gaps_s=0.0,
    )
    sad_wv = parser_with_vocal.get_sad_samples("TEST01", total_samples)["A"]
    assert int(sad_wv.sum().item()) == 500
    assert bool(sad_wv[450].item())
    assert not bool(sad_wv[350].item())

    parser_bridged = AMIWordsXMLParser(
        data_dir=tmp_path,
        sampling_frequency=fs,
        include_vocalsounds=True,
        bridge_gaps_s=0.15,
    )
    sad_bridged = parser_bridged.get_sad_samples("TEST01", total_samples)["A"]
    assert int(sad_bridged.sum().item()) == 700
    assert bool(sad_bridged[100:800].all().item())


def test_dynamic_chunk_dataset_corpus_duration_and_trailing_eval(
    tmp_path: Path,
) -> None:
    """Verifies E25 (100% unique corpus length + weighted sampling) & E28 (short trailing eval chunk)."""
    fs = 16000
    transform = _make_transform(fs=fs)

    # M01 has Array1 + Array2 (unique 4.0 s averaged across arrays), M02 has Array1 (2.0 s)
    train_dir = tmp_path / "train"
    _write_synthetic_precomputed_split(
        train_dir,
        {
            "M01_Array1": 4 * fs,
            "M01_Array2": 4 * fs,
            "M02_Array1": 2 * fs,
        },
    )
    train_ds = DynamicChunkDataset(
        data_dir=train_dir,
        split="train",
        chunk_length_s=2.0,
        min_context_s=1.0,
        train_items_per_epoch=None,
        augmentation={"enabled": False},
        fs=fs,
        transform=transform,
    )
    # 6.0 s unique corpus / 2.0 s chunk = 3 items
    assert train_ds.train_items_per_epoch == 3
    assert len(train_ds) == 3
    w_by_key = dict(zip(train_ds.valid_keys, train_ds.recording_weights))
    assert w_by_key["M01_Array1"] == float(2 * fs + 1)
    assert w_by_key["M02_Array1"] == 1.0

    # Eval trailing chunk retention (>= hop_length) & sub-hop skipping (< hop_length)
    val_dir = tmp_path / "val"
    _write_synthetic_precomputed_split(
        val_dir,
        {
            # 2.5 s -> chunk 1 [0, 32000], chunk 2 [32000, 40000] (tail length 8000 >= 160)
            "REC1_Array1": int(2.5 * fs),
            # 2.0 s + 80 samples with hop=32000 (min_context=0):
            # chunk 1 [0, 32000], next start 32000 has 80 < 160 -> skipped!
            "REC2_Array1": 2 * fs + 80,
        },
    )
    val_ds = DynamicChunkDataset(
        data_dir=val_dir,
        split="val",
        chunk_length_s=2.0,
        min_context_s=0.0,
        train_items_per_epoch=None,
        augmentation={"enabled": False},
        fs=fs,
        transform=transform,
        eval_chunk_length_s=2.0,
        eval_min_context_s=0.0,
    )
    rec1_grid = [entry for entry in val_ds.grid if entry[0] == "REC1_Array1"]
    rec2_grid = [entry for entry in val_ds.grid if entry[0] == "REC2_Array1"]
    assert rec1_grid == [("REC1_Array1", 0, False), ("REC1_Array1", 2 * fs, True)]
    assert rec2_grid == [("REC2_Array1", 0, True)]

    item_tail = val_ds[1]
    meta_tail = item_tail["meta"]
    assert item_tail["input"].shape[-1] == int(0.5 * fs)
    assert meta_tail["start_sample"] == 2 * fs
    assert meta_tail["chunk_samples"] == int(0.5 * fs)
    assert meta_tail["is_last_chunk"] is True


def test_valid_frames_masking_and_variable_length_batches() -> None:
    """Verifies B14 out_norm and E28 valid_frames attention/upsampling masking."""
    torch.manual_seed(7)
    transform = _make_transform()
    logmel = LogMel_Feature_Extractor(transform=transform, n_mels=16)
    csipd = CSIPD_Feature_Extractor(transform=transform)

    t_short = 150
    t_long = 300

    for causal in (False, True):
        est_single = TransformerEstimator(
            input_dim=logmel.feature_dim,
            transform=transform,
            max_sources=4,
            num_input_channels=1,
            causal=causal,
            chunk_size=7,
            subsample=10,
            embed_dim=32,
            n_heads=2,
            ff_inner=64,
            n_layers=2,
            dropout=0.0,
        )
        est_single.eval()
        assert isinstance(est_single.out_norm, nn.LayerNorm)

        x_short = torch.randn(1, 1, logmel.feature_dim, t_short)
        x_long_pad = torch.zeros(2, 1, logmel.feature_dim, t_long)
        x_long_pad[0, :, :, :t_short] = x_short[0]
        x_long_pad[1] = torch.randn(1, logmel.feature_dim, t_long)
        valid_frames = torch.tensor([t_short, t_long], dtype=torch.long)

        with torch.no_grad():
            out_single = est_single(x_short, valid_frames=torch.tensor([t_short]))
            out_batched = est_single(x_long_pad, valid_frames=valid_frames)

        assert out_single.shape == (1, t_short, 5)
        assert out_batched.shape == (2, t_long, 5)
        assert torch.allclose(out_single[0], out_batched[0, :t_short], atol=1e-4)

        for fusion_mode in ("early", "film"):
            est_fusion = FusionTransformerEstimator(
                input_dim=logmel.feature_dim,
                transform=transform,
                max_sources=4,
                cond_input_dim=csipd.feature_dim,
                cond_num_channels=1,
                fusion_mode=fusion_mode,
                num_input_channels=1,
                causal=causal,
                chunk_size=7,
                subsample=10,
                embed_dim=32,
                n_heads=2,
                ff_inner=64,
                n_layers=2,
                dropout=0.0,
            )
            est_fusion.eval()
            assert isinstance(est_fusion.out_norm, nn.LayerNorm)

            c_short = torch.randn(1, 1, csipd.feature_dim, t_short)
            c_long_pad = torch.zeros(2, 1, csipd.feature_dim, t_long)
            c_long_pad[0, :, :, :t_short] = c_short[0]
            c_long_pad[1] = torch.randn(1, csipd.feature_dim, t_long)

            with torch.no_grad():
                f_single = est_fusion(
                    x_short,
                    cond_features=c_short,
                    valid_frames=torch.tensor([t_short]),
                )
                f_batched = est_fusion(
                    x_long_pad,
                    cond_features=c_long_pad,
                    valid_frames=valid_frames,
                )

            assert torch.allclose(f_single[0], f_batched[0, :t_short], atol=1e-4)


def test_osc_stitching_causal_burnin_and_overlap_avg_with_short_tail() -> None:
    """Verifies E28 causal_burnin and overlap_avg handle shorter trailing chunks cleanly."""
    transform = _make_transform()
    logmel = LogMel_Feature_Extractor(transform=transform, n_mels=16)

    for stitch_mode in ("causal_burnin", "overlap_avg"):
        est = TransformerEstimator(
            input_dim=logmel.feature_dim,
            transform=transform,
            max_sources=4,
            num_input_channels=1,
            causal=(stitch_mode == "causal_burnin"),
            chunk_size=5,
            subsample=4,
            embed_dim=16,
            n_heads=2,
            ff_inner=32,
            n_layers=1,
            dropout=0.0,
        )
        module = OSCmodule(
            transform=transform,
            feature_extractor=logmel,
            source_count_estimator=est,
            eval_stitch_mode=stitch_mode,
        )

        # Chunk 1: 20 frames (3200 samples), hop = 10 frames (1600 samples) -> prev_tail = 10 frames
        # Chunk 2 (trailing): 6 frames (960 samples < prev_tail!), starts at 1600 samples
        hop_samples = 10 * transform.hop_length
        chunk1_samples = 20 * transform.hop_length
        chunk2_samples = 6 * transform.hop_length
        max_frames = 20

        estimates = torch.randn(2, max_frames, 5)
        targets = torch.randint(0, 5, (2, max_frames))
        processed_batch = {
            "estimates": estimates,
            "time_lengths": torch.tensor([chunk1_samples, chunk2_samples]),
            "meta": {
                "recording_id": ["REC_A", "REC_A"],
                "start_sample": [0, hop_samples],
                "hop_samples": [hop_samples, hop_samples],
                "is_last_chunk": [False, True],
                "source_count": targets,
            },
        }

        out_ests, out_tgts, _ = module._prepare_eval_estimates_and_targets(
            processed_batch, step_type="test", dataloader_idx=0
        )
        total_emitted = sum(int(t.shape[0]) for t in out_ests)
        if stitch_mode == "causal_burnin":
            # Full recording spans frames [0, 20) from chunk 1 (tail chunk ends at 16 <= 20)
            assert total_emitted == 20
        else:
            # Chunk 1 emits first hop_frames (10) + averaged tail (6) from chunk 2 -> 16 frames total
            assert total_emitted == 16
        assert sum(int(t.shape[0]) for t in out_tgts) == total_emitted
        assert module._eval_overlap_state["test"] is None


def test_radam_param_groups_and_stitched_eval_loss() -> None:
    """Verifies E26 RAdam 2D+ vs <2D weight_decay groups and E27 stitched val/test loss."""
    transform = _make_transform()
    logmel = LogMel_Feature_Extractor(transform=transform, n_mels=16)
    est = TransformerEstimator(
        input_dim=logmel.feature_dim,
        transform=transform,
        max_sources=4,
        num_input_channels=1,
        causal=False,
        chunk_size=5,
        subsample=4,
        embed_dim=16,
        n_heads=2,
        ff_inner=32,
        n_layers=1,
        dropout=0.0,
    )
    module = OSCmodule(
        transform=transform,
        feature_extractor=logmel,
        source_count_estimator=est,
        optimizer_config={
            "RAdam": {
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "decoupled_weight_decay": True,
            }
        },
        eval_stitch_mode="overlap_avg",
    )

    opt_cfg = module.configure_optimizers()
    assert isinstance(opt_cfg, dict) and "optimizer" in opt_cfg
    opt = opt_cfg["optimizer"]
    assert isinstance(opt, torch.optim.RAdam)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 1e-4
    assert all(p.ndim >= 2 for p in opt.param_groups[0]["params"])
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.ndim < 2 for p in opt.param_groups[1]["params"])

    # Verify stitched val/loss and test/loss run and call _prepare_eval_estimates_and_targets once
    hop_samples = 10 * transform.hop_length
    chunk_samples = 20 * transform.hop_length
    n_frames = 20
    batch = {
        "input": torch.randn(2, 1, chunk_samples),
        "mic_lengths": torch.tensor([1, 1]),
        "time_lengths": torch.tensor([chunk_samples, chunk_samples]),
        "meta": {
            "source_count": [
                torch.randint(0, 5, (n_frames,)),
                torch.randint(0, 5, (n_frames,)),
            ],
            "recording_id": ["REC_X", "REC_X"],
            "start_sample": [0, hop_samples],
            "hop_samples": [hop_samples, hop_samples],
            "is_last_chunk": [False, True],
        },
    }

    logged: dict[str, torch.Tensor] = {}

    def _capture_log_dict(
        payload: dict[str, Any], **kwargs: Any
    ) -> None:  # noqa: ANN401
        logged.update(payload)

    module.log_dict = _capture_log_dict  # type: ignore[method-assign]
    module._prepare_eval_estimates_and_targets = MagicMock(  # type: ignore[method-assign]
        wraps=module._prepare_eval_estimates_and_targets
    )

    module.validation_step(batch, 0)
    assert "val/loss" in logged and torch.isfinite(logged["val/loss"])
    assert module._prepare_eval_estimates_and_targets.call_count == 1

    module._prepare_eval_estimates_and_targets.reset_mock()
    module.test_step(batch, 0)
    assert "test/loss" in logged and torch.isfinite(logged["test/loss"])
    assert module._prepare_eval_estimates_and_targets.call_count == 1


def test_metrics_ap_tolerant_event_f1_and_save_results(tmp_path: Path) -> None:
    """Verifies F30 AveragePrecision, full-recording multiplicity TolerantEventF1, and test_events.csv."""
    transform = _make_transform()  # hop = 160 samples -> 0.01 s / frame (100 fps)

    # 1. AveragePrecision: missing class (class 2, 3, 4 absent) should yield 0.0 without NaN
    ap_metric = AveragePrecision(max_sources=4)
    logits = torch.tensor(
        [
            [
                [10.0, -5.0, -5.0, -5.0, -5.0],
                [-5.0, 10.0, -5.0, -5.0, -5.0],
                [10.0, -5.0, -5.0, -5.0, -5.0],
            ]
        ]
    )
    ap_targets = torch.tensor([[0, 1, 0]])
    ap_metric.update([logits[0]], [ap_targets[0]], meta={}, dataloader_idx=0)
    ap_scores = ap_metric.compute()
    assert abs(float(ap_scores["AP_0spk"].item()) - 1.0) < 1e-6
    assert abs(float(ap_scores["AP_1spk"].item()) - 1.0) < 1e-6
    assert float(ap_scores["AP_2spk"].item()) == 0.0
    assert not torch.isnan(ap_scores["mAP"])
    assert ap_metric.get_dataframe() is None

    # 2. TolerantEventF1:
    #    - recA across two chunks [1,1,1] + [1,1,1] -> NO border false event
    #    - recB has 0->2 step at frame 5; pred has 0->1 at frame 10 (+0.05s) & 1->2 at frame 20 (+0.15s)
    #      Both should match the 2 elementary activations from 0->2 within tolerance_time=1.0s!
    ev_metric = TolerantEventF1(transform=transform, tolerance_time=1.0)
    targets_list = [
        torch.tensor([1, 1, 1]),
        torch.tensor([1, 1, 1]),
        torch.cat([torch.zeros(5, dtype=torch.long), torch.full((25,), 2)]),
    ]
    preds_raw_logits = []
    # recA chunk 1 & chunk 2: class 1
    for t_seq in targets_list[:2]:
        lg = torch.full((len(t_seq), 5), -10.0)
        lg[torch.arange(len(t_seq)), t_seq] = 10.0
        preds_raw_logits.append(lg)
    # recB: 0 for frames 0..9, 1 for 10..19, 2 for 20..29
    pred_seq_b = torch.cat(
        [
            torch.zeros(10, dtype=torch.long),
            torch.ones(10, dtype=torch.long),
            torch.full((10,), 2, dtype=torch.long),
        ]
    )
    lg_b = torch.full((30, 5), -10.0)
    lg_b[torch.arange(30), pred_seq_b] = 10.0
    preds_raw_logits.append(lg_b)

    meta = {"recording_id": ["recA", "recA", "recB"]}
    ev_metric.update(preds_raw_logits, targets_list, meta=meta, dataloader_idx=0)
    ev_out = ev_metric.compute()

    assert abs(float(ev_out["F1score_act"].item()) - 1.0) < 1e-6
    assert abs(float(ev_out["F1score_overall"].item()) - 1.0) < 1e-6
    # Delays: (10 - 5)*0.01 = +0.05 s and (20 - 5)*0.01 = +0.15 s
    # torch.Tensor.median() selects the lower middle value on even-length inputs (0.05 s)
    assert abs(float(ev_out["delay_mean_act"].item()) - 0.10) < 1e-5
    assert abs(float(ev_out["delay_median_act"].item()) - 0.05) < 1e-5
    assert abs(float(ev_out["delay_iqr_act"].item()) - 0.05) < 1e-5
    assert abs(float(ev_out["delay_abs_mean_act"].item()) - 0.10) < 1e-5
    assert ev_metric.get_dataframe() is None

    events_df = ev_metric.get_events_dataframe()
    assert isinstance(events_df, pd.DataFrame)
    assert len(events_df) == 2
    assert set(events_df["recording_id"]) == {"recB"}
    assert set(events_df["status"]) == {"TP"}
    assert list(events_df.columns) == [
        "recording_id",
        "event_type",
        "status",
        "true_frame",
        "pred_frame",
        "true_time_s",
        "pred_time_s",
        "delay_s",
        "true_step",
        "true_raw_jump",
        "pred_step",
        "pred_raw_jump",
        "num_mics",
    ]

    # 3. SaveTestResultsCallback writes test_events.csv (and test_events_<ds>.csv) alongside summary CSV
    logmel = LogMel_Feature_Extractor(transform=transform, n_mels=16)
    est = TransformerEstimator(
        input_dim=logmel.feature_dim,
        transform=transform,
        max_sources=4,
        num_input_channels=1,
        causal=False,
        chunk_size=5,
        subsample=4,
        embed_dim=16,
        n_heads=2,
        ff_inner=32,
        n_layers=1,
        dropout=0.0,
    )
    osc = OSCmodule(
        transform=transform,
        feature_extractor=logmel,
        source_count_estimator=est,
        metrics_test={
            "AveragePrecision": {"max_sources": 4},
            "TolerantEventF1": {"tolerance_time": 1.0},
        },
    )
    osc.log_dict = MagicMock()  # type: ignore[method-assign]
    osc.metric_collections["test"].update(
        [lg_b], [targets_list[2]], {"recording_id": ["recB"]}, 0
    )

    save_dir = tmp_path / "results_single"
    callback = SaveTestResultsCallback(save_dir=str(save_dir))
    trainer_mock = MagicMock()
    trainer_mock.callback_metrics = {"test/loss": torch.tensor(0.1)}
    callback.on_test_epoch_end(trainer_mock, osc)

    assert (save_dir / "test_metrics_summary.csv").exists()
    assert (save_dir / "test_events.csv").exists()
    saved_events = pd.read_csv(save_dir / "test_events.csv")
    assert len(saved_events) == 2

    # Multi-dataset per-dataset test_events_<ds>.csv verification
    osc.configure_test_datasets(["ami_eval", "brudex_eval"])
    osc.test_metric_collections["ami_eval"].update(
        [lg_b], [targets_list[2]], {"recording_id": ["recB"]}, 0
    )
    osc.test_metric_collections["brudex_eval"].update(
        [lg_b], [targets_list[2]], {"recording_id": ["recC"]}, 1
    )
    osc.metric_collections["test"].update(
        [lg_b], [targets_list[2]], {"recording_id": ["recB"]}, 0
    )
    osc.metric_collections["test"].update(
        [lg_b], [targets_list[2]], {"recording_id": ["recC"]}, 1
    )
    save_dir_multi = tmp_path / "results_multi"
    SaveTestResultsCallback(save_dir=str(save_dir_multi)).on_test_epoch_end(
        trainer_mock, osc
    )
    assert (save_dir_multi / "test_events_ami_eval.csv").exists()
    assert (save_dir_multi / "test_events_brudex_eval.csv").exists()
    assert (save_dir_multi / "test_events.csv").exists()


def test_tolerant_event_f1_causal_and_non_causal_modes() -> None:
    """Verifies TolerantEventF1 symmetric, strict causal, and asymmetric matching modes."""
    transform = _make_transform()  # 10 ms per frame -> 1 frame == 0.01 s

    # True activation at frame 10 (0.10 s), predicted early at frame 7 (0.07 s, -0.03 s offset)
    true_seq = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    pred_seq = torch.cat([torch.zeros(7, dtype=torch.long), torch.ones(23, dtype=torch.long)])
    logits = torch.full((30, 5), -10.0)
    logits[torch.arange(30), pred_seq] = 10.0

    # 1. Non-causal mode matches early detection within [-1.0, +1.0] s
    ev_noncausal = TolerantEventF1(
        transform=transform,
        tolerance_time=1.0,
        past_tolerance_time=0.0,
        causal_mode=False,
    )
    ev_noncausal.update([logits], [true_seq], meta={"recording_id": ["rec1"]}, dataloader_idx=0)
    res_nc = ev_noncausal.compute()
    assert abs(float(res_nc["F1score_act"].item()) - 1.0) < 1e-5
    assert abs(float(res_nc["delay_mean_act"].item()) - (-0.03)) < 1e-5

    # 2. Strictly causal mode with past_tolerance_time=0.0 rejects premature detection
    ev_causal = TolerantEventF1(
        transform=transform,
        tolerance_time=1.0,
        past_tolerance_time=0.0,
        causal_mode=True,
    )
    ev_causal.update([logits], [true_seq], meta={"recording_id": ["rec1"]}, dataloader_idx=0)
    res_c = ev_causal.compute()
    assert float(res_c["F1score_act"].item()) == 0.0
    events_df = ev_causal.get_events_dataframe()
    assert events_df is not None
    assert set(events_df["status"]) == {"FP", "FN"}

    # 3. Causal mode with bounded look-ahead jitter past_tolerance_time=0.05 accepts -0.03 s
    ev_causal_tol = TolerantEventF1(
        transform=transform,
        tolerance_time=1.0,
        past_tolerance_time=0.05,
        causal_mode=True,
    )
    ev_causal_tol.update([logits], [true_seq], meta={"recording_id": ["rec1"]}, dataloader_idx=0)
    res_ct = ev_causal_tol.compute()
    assert abs(float(res_ct["F1score_act"].item()) - 1.0) < 1e-5
    assert abs(float(res_ct["delay_mean_act"].item()) - (-0.03)) < 1e-5

    # 4. OSCmodule auto-resolves TolerantEventF1 causality via estimator_causal
    logmel = LogMel_Feature_Extractor(transform=transform, n_mels=16)
    for is_causal in (False, True):
        est = TransformerEstimator(
            input_dim=logmel.feature_dim,
            transform=transform,
            max_sources=4,
            num_input_channels=1,
            causal=is_causal,
            chunk_size=5,
            subsample=4,
            embed_dim=16,
            n_heads=2,
            ff_inner=32,
            n_layers=1,
            dropout=0.0,
        )
        osc = OSCmodule(
            transform=transform,
            feature_extractor=logmel,
            source_count_estimator=est,
            metrics_test={
                "TolerantEventF1": {
                    "tolerance_time": 1.0,
                    "past_tolerance_time": 0.0,
                    "causal_mode": "auto",
                }
            },
        )
        t_metric = osc.metric_collections["test"]["TolerantEventF1"]
        assert t_metric.is_causal_eval is is_causal
        assert t_metric.past_tolerance == (0 if is_causal else 100)


def test_ami_segments_default_and_parser_cache_invalidation(tmp_path: Path) -> None:
    """Verifies YAML defaults ([2400, 300, 300], AMISegmentsXMLParser) and SAD cache refresh."""
    repo_root = Path("/data4/Henri/MuSE-Toolbox")
    sim_cfg = yaml.safe_load(
        (repo_root / "configs/dataset/base_simulated_dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert sim_cfg["num_scenarios"] == [2400, 300, 300]

    ami_cfg = yaml.safe_load(
        (repo_root / "configs/dataset/ami.yaml").read_text(encoding="utf-8")
    )
    assert (
        ami_cfg["label_parser"]["_target_"]
        == "muse_toolbox.data.databases.ami_label_parsers.AMISegmentsXMLParser"
    )

    fs = 16000
    num_samples = fs  # 1.0 s
    meeting_id = "EN2001a"
    raw_dir = tmp_path / "raw_ami"
    precomputed_dir = tmp_path / "precomputed"
    audio_dir = raw_dir / meeting_id / "audio"
    audio_dir.mkdir(parents=True)
    words_dir = raw_dir / "Annotations_1.6.2_manual" / "words"
    words_dir.mkdir(parents=True)
    seg_dir = raw_dir / "Annotations_1.6.2_manual" / "segments"
    seg_dir.mkdir(parents=True)

    sf.write(
        str(audio_dir / f"{meeting_id}.Array1-01.wav"),
        np.zeros((num_samples,), dtype=np.float32),
        fs,
    )
    (words_dir / f"{meeting_id}.A.words.xml").write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
        '<w starttime="0.10" endtime="0.20">hi</w>'
        "</nite:root>",
        encoding="utf-8",
    )
    (seg_dir / f"{meeting_id}.A.segments.xml").write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
        '<segment transcriber_start="0.10" transcriber_end="0.50"/>'
        "</nite:root>",
        encoding="utf-8",
    )

    transform = _make_transform(fs=fs)
    common_dm_kwargs: dict[str, Any] = {
        "train_meetings": [meeting_id],
        "val_meetings": [],
        "test_meetings": [],
        "chunk_length_s": 1.0,
        "min_context_s": 0.0,
        "train_items_per_epoch": None,
        "arrays": ["Array1"],
        "augmentation": {"enabled": False},
        "sampling_frequency": fs,
        "data_dir": str(precomputed_dir),
        "precomputed_dir": str(precomputed_dir),
        "transform": transform,
        "batch_size": 1,
        "num_workers": 0,
        "generation_config": {"max_sources": 4},
        "seed": 42,
        "reset": False,
        "acc_device": "cpu",
    }

    word_parser = AMIWordsXMLParser(data_dir=raw_dir, sampling_frequency=fs)
    db_words = AMIDatabase(
        data_dir=raw_dir, label_parser=word_parser, sampling_frequency=fs
    )
    dm_words = AMIDataModule(database=db_words, **common_dm_kwargs)
    dm_words.prepare_data()

    audio_npy = dm_words.precomputed_dir / "train" / f"{meeting_id}_Array1_audio.npy"
    sad_npy = dm_words.precomputed_dir / "train" / f"{meeting_id}_Array1_sad.npy"
    assert int(np.load(sad_npy).sum()) == int(0.10 * fs)
    orig_audio_mtime = audio_npy.stat().st_mtime_ns

    seg_parser = AMISegmentsXMLParser(data_dir=raw_dir, sampling_frequency=fs)
    db_segments = AMIDatabase(
        data_dir=raw_dir, label_parser=seg_parser, sampling_frequency=fs
    )
    db_segments.get_meeting = MagicMock(  # type: ignore[method-assign]
        wraps=db_segments.get_meeting
    )
    dm_segments = AMIDataModule(database=db_segments, **common_dm_kwargs)
    dm_segments.prepare_data()

    # SAD regenerated from segments.xml (0.40 s) while audio WAV loading was skipped
    assert db_segments.get_meeting.call_count == 0
    assert audio_npy.stat().st_mtime_ns == orig_audio_mtime
    assert int(np.load(sad_npy).sum()) == int(0.40 * fs)


def test_ap_vad_osd_and_num_mics_stratified_csv(tmp_path: Path) -> None:
    """Verifies AP_VAD/AP_OSD (#11) and per-num_mics CSV stratification (#16)."""
    transform = _make_transform()
    ap = AveragePrecision(max_sources=4)

    # Frame targets: 0 (noise), 1 (1-spk), 2 (2-spk), 3 (3-spk)
    logits = torch.tensor(
        [
            [5.0, -5.0, -5.0, -5.0, -5.0],  # 0
            [-5.0, 5.0, -5.0, -5.0, -5.0],  # 1
            [-5.0, -5.0, 5.0, -5.0, -5.0],  # 2
            [-5.0, -5.0, -5.0, 5.0, -5.0],  # 3
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    ap.update([logits], [targets], meta={"num_mics": [3]}, dataloader_idx=0)
    # Add a 6-mic chunk with imperfect VAD/OSD separation
    logits_6 = torch.tensor(
        [
            [2.0, 1.0, -5.0, -5.0, -5.0],
            [-5.0, 4.0, 1.0, -5.0, -5.0],
            [-5.0, -5.0, 4.0, -5.0, -5.0],
        ],
        dtype=torch.float32,
    )
    targets_6 = torch.tensor([1, 0, 2], dtype=torch.long)
    ap.update([logits_6], [targets_6], meta={"num_mics": [6]}, dataloader_idx=0)

    out = ap.compute()
    assert "AP_VAD" in out and "AP_OSD" in out
    by_nm = ap.compute_by_num_mics()
    assert set(by_nm.keys()) == {3, 6}
    assert abs(by_nm[3]["AP_VAD"] - 1.0) < 1e-6
    assert abs(by_nm[3]["AP_OSD"] - 1.0) < 1e-6
    assert abs(by_nm[3]["mAP"] - 1.0) < 1e-6

    # End-to-end SaveTestResultsCallback writes test_metrics.csv and test_metrics_by_num_mics.csv
    logmel = LogMel_Feature_Extractor(transform=transform, n_mels=16)
    est = TransformerEstimator(
        input_dim=logmel.feature_dim,
        transform=transform,
        max_sources=4,
        num_input_channels=1,
        causal=False,
        chunk_size=5,
        subsample=4,
        embed_dim=16,
        n_heads=2,
        ff_inner=32,
        n_layers=1,
        dropout=0.0,
    )
    osc = OSCmodule(
        transform=transform,
        feature_extractor=logmel,
        source_count_estimator=est,
        metrics_test={
            "AveragePrecision": {"max_sources": 4},
            "TolerantEventF1": {"tolerance_time": 1.0},
            "META2DF": {},
            "Accuracy": {},
            "MAE": {},
        },
    )
    osc.log_dict = MagicMock()  # type: ignore[method-assign]
    osc.metric_collections["test"].update(
        [logits, logits_6],
        [targets, targets_6],
        {
            "scenario_id": ["sc_3m_0.00", "sc_6m_0.00"],
            "recording_id": ["sc_3m", "sc_6m"],
            "num_mics": [3, 6],
            "mic_indices": [[0, 2, 5], [0, 1, 2, 3, 4, 7]],
            "scenario_params": [{"num_mics": 3}, {"num_mics": 6}],
        },
        0,
    )
    save_dir = tmp_path / "nm_results"
    cb = SaveTestResultsCallback(save_dir=str(save_dir))
    trainer_mock = MagicMock()
    trainer_mock.callback_metrics = {"test/loss": torch.tensor(0.2)}
    cb.on_test_epoch_end(trainer_mock, osc)

    metrics_df = pd.read_csv(save_dir / "test_metrics.csv")
    assert "num_mics" in metrics_df.columns
    assert set(metrics_df["num_mics"].tolist()) == {3, 6}

    by_nm_df = pd.read_csv(save_dir / "test_metrics_by_num_mics.csv")
    assert by_nm_df["num_mics"].tolist() == [3, 6]
    assert "AP_VAD" in by_nm_df.columns and "AP_OSD" in by_nm_df.columns
    assert "F1score_overall" in by_nm_df.columns


def test_additive_170_epoch_and_length_weighted_aug(tmp_path: Path) -> None:
    """Verifies +70% additive epoch length (#13) and length-weighted aug sampling (#12)."""
    fs = 16000
    transform = _make_transform(fs=fs)
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)

    # Create two single-speaker recordings: LONG (20 s) and SHORT (2 s), each with 1 active speaker
    for key, dur_s, spk_row in (("LONG_Array1", 20, 0), ("SHORT_Array1", 2, 0)):
        n_samp = dur_s * fs
        audio = np.random.default_rng(1).normal(0.0, 0.05, size=(2, n_samp)).astype(
            np.float32
        )
        sad = np.zeros((2, n_samp), dtype=bool)
        sad[spk_row, :] = True
        np.save(split_dir / f"{key}_audio.npy", audio)
        np.save(split_dir / f"{key}_sad.npy", sad)
        torch.save(
            {"meeting_id": key.split("_")[0], "array_id": "Array1", "sampling_frequency": fs},
            split_dir / f"{key}_meta.pt",
        )
    np.save(
        tmp_path / f"single_speaker_index_{fs}Hz.npy",
        np.array(["LONG_Array1_0", "SHORT_Array1_0"], dtype=str),
    )

    ds = DynamicChunkDataset(
        data_dir=split_dir,
        split="train",
        chunk_length_s=2.0,
        min_context_s=1.0,
        train_items_per_epoch=10,
        augmentation={
            "enabled": True,
            "augmentation_factor": 0.70,
            "prob_1spk": 1.0,
            "prob_2spk": 0.0,
            "prob_3spk": 0.0,
            "prob_4spk": 0.0,
        },
        fs=fs,
        transform=transform,
    )
    # 10 base items + round(10 * 0.70) = 17 total items per epoch
    assert ds.base_train_items_per_epoch == 10
    assert ds.num_augmented_items == 7
    assert len(ds) == 17
    assert ds[0]["meta"]["scenario_params"]["augmentation_applied"] is False
    assert ds[9]["meta"]["scenario_params"]["augmentation_applied"] is False
    assert ds[10]["meta"]["scenario_params"]["augmentation_applied"] is True
    assert ds[16]["meta"]["scenario_params"]["augmentation_applied"] is True

    # Length-weighted selection: LONG (20s) should dominate SHORT (2s) across aug draws
    long_hits = 0
    for j in range(60):
        rec_id = ds[10 + (j % 7)]["meta"]["recording_id"]
        if "LONG_" in rec_id:
            long_hits += 1
    assert long_hits > 40, f"Expected length-weighted LONG > 40/60, got {long_hits}"


def test_dynamic_mic_subsampling_rand3_8_and_single_channel_expansion(
    tmp_path: Path,
) -> None:
    """Verifies dynamic Rand3-8 (#16), config removal of Circ3-8, and 1-ch train/val/test (#6)."""
    import random as py_random
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from muse_toolbox.data.components.collate import raw_audio_collate_fn

    repo_root = Path(__file__).resolve().parent.parent
    assert not (repo_root / "configs/dataset/pra_anf_circ_2-8ch.yaml").exists()
    assert not (repo_root / "configs/dataset/pra_anf_rand_2-8ch.yaml").exists()
    assert (repo_root / "configs/dataset/pra_anf_rand_3-8ch.yaml").exists()

    # Check Rand3-8 Hydra config points to Rand8 precomputed_id & uniform 3..8 probs
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(repo_root / "configs"), version_base=None
    ):
        cfg_r38 = compose(
            config_name="default", overrides=["dataset=pra_anf_rand_3-8ch"]
        )
    assert cfg_r38.dataset.precomputed_id == "J3_PRAANF_rand_8ch_20cm_SIR10dB"
    assert set(int(k) for k in cfg_r38.dataset.mic_count_probs.keys()) == {
        3,
        4,
        5,
        6,
        7,
        8,
    }

    fs = 16000
    transform = _make_transform(fs=fs)
    base_dir = tmp_path / "J3_PRAANF_rand_8ch_20cm_SIR10dB"
    for sp in ("train", "val", "test"):
        sp_dir = base_dir / sp
        sp_dir.mkdir(parents=True)
        for key in ("SC0_Array1", "SC1_Array1", "SC2_Array1"):
            n_samp = 5 * fs
            audio = np.random.default_rng(7).normal(0.0, 0.05, size=(8, n_samp)).astype(
                np.float32
            )
            sad = np.zeros((2, n_samp), dtype=bool)
            sad[0, : n_samp // 2] = True
            np.save(sp_dir / f"{key}_audio.npy", audio)
            np.save(sp_dir / f"{key}_sad.npy", sad)
            torch.save(
                {
                    "meeting_id": key.split("_")[0],
                    "array_id": "Array1",
                    "sampling_frequency": fs,
                    "scenario_params": {
                        "num_sources": 1,
                        "mic_pos": np.arange(24, dtype=np.float32).reshape(3, 8),
                    },
                },
                sp_dir / f"{key}_meta.pt",
            )

    # Validation errors on invalid mic_count_probs
    for bad_probs in ({0: 1.0}, {9: 1.0}, {3: 0.5, 4: 0.1}):
        try:
            DynamicChunkDataset(
                data_dir=base_dir / "val",
                split="val",
                chunk_length_s=3.0,
                min_context_s=1.5,
                train_items_per_epoch=None,
                augmentation={"enabled": False},
                fs=fs,
                transform=transform,
                mic_count_probs=bad_probs,
            )
            raise AssertionError(f"Expected ValueError for bad_probs={bad_probs}")
        except ValueError:
            pass

    # Deterministic per-recording (M, mic_indices) on val across sliding chunks of same recording
    val_ds = DynamicChunkDataset(
        data_dir=base_dir / "val",
        split="val",
        chunk_length_s=3.0,
        min_context_s=1.5,
        train_items_per_epoch=None,
        augmentation={"enabled": False},
        fs=fs,
        transform=transform,
        eval_chunk_length_s=3.0,
        eval_min_context_s=1.5,
        mic_count_probs={3: 1 / 6, 4: 1 / 6, 5: 1 / 6, 6: 1 / 6, 7: 1 / 6, 8: 1 / 6},
        seed=777,
    )
    by_rec: dict[str, list[tuple[int, list[int]]]] = {}
    for i in range(len(val_ds)):
        item = val_ds[i]
        m = item["meta"]
        nm = m["num_mics"]
        assert 3 <= nm <= 8
        assert item["input"].shape[0] == nm
        assert m["scenario_params"]["num_sources"] == 1  # Not overwritten!
        assert np.asarray(m["scenario_params"]["mic_pos"]).shape == (3, nm)
        by_rec.setdefault(m["recording_id"], []).append((nm, m["mic_indices"]))
    for entries in by_rec.values():
        assert len({tuple(e[1]) for e in entries}) == 1

    # Single-channel expansion: train random mic, val 1 deterministic-random mic/rec, test all M mics
    sc_train = DynamicChunkDataset(
        data_dir=base_dir / "train",
        split="train",
        chunk_length_s=3.0,
        min_context_s=1.5,
        train_items_per_epoch=None,
        augmentation={"enabled": False},
        fs=fs,
        transform=transform,
        single_channel_expansion=True,
        seed=777,
    )
    py_random.seed(123)
    train_mics = {sc_train[i]["meta"]["mic_indices"][0] for i in range(len(sc_train))}
    assert len(train_mics) > 1
    assert all(sc_train[i]["input"].shape[0] == 1 for i in range(len(sc_train)))

    sc_val = DynamicChunkDataset(
        data_dir=base_dir / "val",
        split="val",
        chunk_length_s=3.0,
        min_context_s=1.5,
        train_items_per_epoch=None,
        augmentation={"enabled": False},
        fs=fs,
        transform=transform,
        eval_chunk_length_s=3.0,
        eval_min_context_s=1.5,
        single_channel_expansion=True,
        seed=777,
    )
    val_rec_ids = {sc_val[i]["meta"]["recording_id"] for i in range(len(sc_val))}
    # Exactly 1 channel per original recording in val (3 recordings -> 3 val recording_ids)
    assert len(val_rec_ids) == 3

    sc_test = DynamicChunkDataset(
        data_dir=base_dir / "test",
        split="test",
        chunk_length_s=3.0,
        min_context_s=1.5,
        train_items_per_epoch=None,
        augmentation={"enabled": False},
        fs=fs,
        transform=transform,
        eval_chunk_length_s=3.0,
        eval_min_context_s=1.5,
        single_channel_expansion=True,
        seed=777,
    )
    test_rec_ids = {sc_test[i]["meta"]["recording_id"] for i in range(len(sc_test))}
    # Exhaustive across all 8 channels for all 3 recordings -> 24 recording_ids
    assert len(test_rec_ids) == 24
    assert all(sc_test[i]["input"].shape[0] == 1 for i in range(len(sc_test)))
    _ = raw_audio_collate_fn([sc_test[0], sc_test[1]])


def test_predict_stitching_and_ckpt_selection(tmp_path: Path) -> None:
    """Verifies OSCmodule predict_step stitching (#17) and _resolve_eval_ckpt_path (#1)."""
    from omegaconf import OmegaConf
    from muse_toolbox.pipelines.source_counting_pipeline import _resolve_eval_ckpt_path

    # 1. Checkpoint selection resolution (#1)
    cfg_train = OmegaConf.create({"train": True})
    trainer_with_ckpt = MagicMock()
    trainer_with_ckpt.checkpoint_callback.best_model_path = "/tmp/best.ckpt"
    assert _resolve_eval_ckpt_path(cfg_train, trainer_with_ckpt) == "best"

    trainer_no_ckpt = MagicMock()
    trainer_no_ckpt.checkpoint_callback.best_model_path = ""
    assert _resolve_eval_ckpt_path(cfg_train, trainer_no_ckpt) is None

    # 2. Stitched predict_step matches test_step continuous length
    transform = _make_transform()
    logmel = LogMel_Feature_Extractor(transform=transform, n_mels=16)
    est = TransformerEstimator(
        input_dim=logmel.feature_dim,
        transform=transform,
        max_sources=4,
        num_input_channels=1,
        causal=False,
        chunk_size=3,
        subsample=5,
        embed_dim=16,
        n_heads=2,
        ff_inner=32,
        n_layers=1,
        dropout=0.0,
    )
    osc = OSCmodule(
        transform=transform,
        feature_extractor=logmel,
        source_count_estimator=est,
    )
    osc.eval()
    osc.on_predict_epoch_start()

    # Two overlapping 3.0 s chunks (300 frames each) with 1.5 s hop (150 frames): total = 450 frames
    chunks = []
    for pos, (start_samp, is_last) in enumerate(((0, False), (24000, True))):
        batch = {
            "input": torch.randn(1, 2, 48000),
            "time_lengths": torch.tensor([48000], dtype=torch.long),
            "meta": {
                "recording_id": ["rec_pred_0"],
                "scenario_id": [f"rec_pred_0_{pos * 1.5:.2f}"],
                "start_sample": [start_samp],
                "hop_samples": [24000],
                "is_last_chunk": [is_last],
            },
        }
        out = osc.predict_step(batch, batch_idx=pos)
        chunks.extend(out["estimates"])
    osc.on_predict_epoch_end()

    stitched = torch.cat(chunks, dim=0)
    assert stitched.shape == (450,)
    assert stitched.dtype == torch.int64
    _ = tmp_path
