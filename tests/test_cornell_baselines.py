import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

from muse_toolbox.models.components.feature_extractors import (
    CSIPD_Feature_Extractor,
    GMSC_Feature_Extractor,
    LogMel_Feature_Extractor,
    PSC_Feature_Extractor,
    PureSTFTFeatureExtractor,
    WGMSC_Feature_Extractor,
)
from muse_toolbox.models.source_counting.estimators import (
    FiLMLayer,
    FusionTransformerEstimator,
    TransformerEstimator,
)
from muse_toolbox.utils import STFTtransform


def _make_batch(batch_size: int = 2, num_mics: int = 8, num_samples: int = 48000) -> dict[str, Any]:
    """Builds a synthetic raw audio dictionary batch."""
    num_frames = num_samples // 160
    return {
        "input": torch.randn(batch_size, num_mics, num_samples),
        "mic_lengths": torch.full((batch_size,), num_mics, dtype=torch.long),
        "time_lengths": torch.full((batch_size,), num_samples, dtype=torch.long),
        "meta": {
            "source_count": [
                torch.randint(0, 5, (num_frames,), dtype=torch.long)
                for _ in range(batch_size)
            ],
            "scenario_id": [f"sc_{i}" for i in range(batch_size)],
            "scenario_params": [{"num_sources": 2} for _ in range(batch_size)],
        },
    }


def test_output_channels_and_no_channel_combinator() -> None:
    """Verifies output_channels(num_mics) across extractors and removal of channel_combinator."""
    tr = STFTtransform(
        sampling_frequency=16000,
        frame_length=0.025,
        frame_shift=0.010,
        remove_DC=True,
        remove_Nyquist=True,
    )
    assert LogMel_Feature_Extractor(tr, mode="ref").output_channels(None) == 1
    assert LogMel_Feature_Extractor(tr, mode="all").output_channels(8) == 8
    assert PureSTFTFeatureExtractor(tr, mode="ref").output_channels(None) == 1
    assert PureSTFTFeatureExtractor(tr, mode="all").output_channels(6) == 6
    assert GMSC_Feature_Extractor(tr, smoothing_time_constant=0.1).output_channels(None) == 1
    assert WGMSC_Feature_Extractor(
        tr,
        smoothing_time_constant=0.1,
        whitening_time_constant=0.1,
        smoothing_time_constant_rev=0.1,
        whitening_time_constant_rev=0.1,
    ).output_channels(None) == 1
    assert CSIPD_Feature_Extractor(tr, mode="all").output_channels(8) == 28
    assert CSIPD_Feature_Extractor(tr, mode="ref").output_channels(8) == 7
    assert PSC_Feature_Extractor(tr, mode="all").output_channels(8) == 28

    try:
        CSIPD_Feature_Extractor(tr, mode="all").output_channels(None)
        raise AssertionError("Expected ValueError when num_mics=None for CSIPD")
    except ValueError:
        pass


def test_film_layer_initialization_and_gradients() -> None:
    """Verifies FiLMLayer identity-centered initialization (~[0.9, 1.1] scale, [-0.1, 0.1] shift)."""
    torch.manual_seed(0)
    cond_dim = 15 * 28 * 398
    embed_dim = 64
    film = FiLMLayer(cond_dim=cond_dim, embed_dim=embed_dim)

    assert (film.gamma_proj.bias >= -0.1).all() and (film.gamma_proj.bias <= 0.1).all()
    assert (film.beta_proj.bias >= -0.1).all() and (film.beta_proj.bias <= 0.1).all()
    # Ensure distinct channel biases (not all identical)
    assert film.gamma_proj.bias.std().item() > 0.01

    h = torch.randn(2, embed_dim, 12, requires_grad=True)
    cond = torch.randn(2, cond_dim, 12)
    scale = 1.0 + film.gamma_proj(cond)
    assert scale.min().item() > 0.8 and scale.max().item() < 1.2

    out = film(h, cond)
    assert out.shape == h.shape
    out.sum().backward()
    assert film.gamma_proj.weight.grad is not None and film.gamma_proj.weight.grad.abs().sum() > 0
    assert film.beta_proj.weight.grad is not None and film.beta_proj.weight.grad.abs().sum() > 0


def test_three_cornell_baselines_and_causality() -> None:
    """Verifies Baseline 1 (LogMel), Baseline 2 (Early Fusion), and Baseline 3 (Late FiLM)."""
    config_dir = str(PROJECT_ROOT / "configs")
    batch = _make_batch(batch_size=2, num_mics=8, num_samples=48000)

    # --- Baseline 1: Spectral-Only LogMel (default osc.yaml) ---
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_b1 = compose(config_name="default")
    assert "channel_combinator" not in cfg_b1.model
    model_b1 = instantiate(cfg_b1.model)
    assert isinstance(model_b1.source_count_estimator, TransformerEstimator)
    assert model_b1.source_count_estimator.flat_input_dim == 80
    out_b1 = model_b1(batch)["estimates"]
    assert out_b1.shape == (2, 300, 5)
    loss_b1 = model_b1.training_step(batch, idx=0)
    loss_b1.backward()
    assert torch.isfinite(loss_b1)

    # --- Spatial-Only Flattened CSIPD (Option B stem flattening) ---
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_spat = compose(
            config_name="default",
            overrides=[
                "model/feature_extractor=csipd",
                "model.num_mics=8",
            ],
        )
    model_spat = instantiate(cfg_spat.model)
    assert model_spat.source_count_estimator.num_input_channels == 28
    assert model_spat.source_count_estimator.flat_input_dim == 28 * 2 * 199
    out_spat = model_spat(batch)["estimates"]
    assert out_spat.shape == (2, 300, 5)

    # Mismatched mic count must fail clearly in fixed-width baseline estimator
    bad_batch = _make_batch(batch_size=1, num_mics=6, num_samples=16000)
    try:
        model_spat(bad_batch)
        raise AssertionError("Expected ValueError on mismatched num_mics")
    except ValueError:
        pass

    # --- Baseline 2 (Early Fusion) & Baseline 3 (Late FiLM Fusion) ---
    for fusion_mode in ("early", "film"):
        for causal in (False, True):
            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg_f = compose(
                    config_name="default",
                    overrides=[
                        "model=osc_fusion",
                        f"model.source_count_estimator.fusion_mode={fusion_mode}",
                        f"model.source_count_estimator.causal={str(causal).lower()}",
                        "model.source_count_estimator.embed_dim=64",
                        "model.source_count_estimator.ff_inner=128",
                        "model.source_count_estimator.n_layers=2",
                    ],
                )
            model_f = instantiate(cfg_f.model)
            est = model_f.source_count_estimator
            assert isinstance(est, FusionTransformerEstimator)
            assert est.fusion_mode == fusion_mode
            assert est.num_input_channels == 1
            assert est.cond_num_channels == 28
            assert est.flat_input_dim == 80
            assert est.cond_flat_dim == 28 * 2 * 199
            if fusion_mode == "early":
                assert est.film_layers is None
                assert est.bottleneck[0].in_channels == (
                    est.spec_stem.out_dim + est.cond_stem.out_dim
                )
            else:
                assert est.film_layers is not None and len(est.film_layers) == 2
                assert est.bottleneck[0].in_channels == est.spec_stem.out_dim

            # Verify shared single STFT encode call during forward
            with patch.object(
                model_f.transform, "encode", wraps=model_f.transform.encode
            ) as mock_encode:
                out_f = model_f(batch)["estimates"]
                assert mock_encode.call_count == 1
            assert out_f.shape == (2, 300, 5)
            assert torch.isfinite(out_f).all()

            # Verify loss and gradients flow through both stems and FiLM layers
            model_f.train()
            model_f.zero_grad()
            loss_f = model_f.training_step(batch, idx=0)
            assert torch.isfinite(loss_f)
            loss_f.backward()
            assert est.spec_stem.norm.weight.grad is not None
            assert est.cond_stem.norm.weight.grad is not None
            if fusion_mode == "film":
                assert est.film_layers is not None
                for fl in est.film_layers:
                    assert fl.gamma_proj.weight.grad is not None
                    assert fl.beta_proj.weight.grad is not None

            # Verify strict frame-level causality when causal=True
            if causal:
                model_f.eval()
                b_clean = _make_batch(batch_size=1, num_mics=8, num_samples=32000)
                b_pert = {
                    **b_clean,
                    "input": b_clean["input"].clone(),
                }
                # Perturb audio strictly after sample 16000 (frame 100)
                b_pert["input"][:, :, 16000:] += 50.0
                with torch.no_grad():
                    y_clean = model_f(b_clean)["estimates"]
                    y_pert = model_f(b_pert)["estimates"]
                # Frames 0..99 use samples strictly < 16000 under causal left-padded STFT
                assert torch.allclose(y_clean[:, :100, :], y_pert[:, :100, :], atol=1e-5)
