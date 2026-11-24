import functools
import logging

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from muse_toolbox.models.components.feature_extractors import (
    CSIPD_Feature_Extractor,
    PSC_Feature_Extractor,
    PureSTFTFeatureExtractor,
    StackedFeatureExtractor,
)
from muse_toolbox.utils import (
    STFTtransform,
    coherenceMatrix,
    covariance_SCM,
    exp_windowing_recursive,
    gmsc,
)

log = logging.getLogger(__name__)


def _make_stft_transform() -> STFTtransform:
    """Builds a standard STFTtransform matching Cornell defaults."""
    return STFTtransform(
        sampling_frequency=16000,
        frame_length=0.025,
        frame_shift=0.010,
        remove_DC=True,
        remove_Nyquist=True,
    )


def test_coherence_matrix_standard_and_edge_cases() -> None:
    """Verifies coherenceMatrix on normal PSD, silent, zero-padded, and drift cases."""
    torch.manual_seed(0)
    # 1. Standard non-zero complex STFT -> rank-1 SCM coherence
    y = torch.randn(2, 4, 3, 1, dtype=torch.complex64)
    cov = covariance_SCM(y)
    coh = coherenceMatrix(cov)

    assert coh.shape == (2, 4, 3, 3)
    assert torch.isfinite(coh).all()
    assert torch.allclose(coh, coh.mH, atol=1e-6)
    diag = coh.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag.real, torch.ones_like(diag.real), atol=1e-6)
    assert torch.allclose(diag.imag, torch.zeros_like(diag.imag), atol=1e-6)
    assert (coh.abs() <= 1.0 + 1e-6).all()

    # 2. All-zero silent covariance -> Identity matrix (gmsc == 0, no NaNs)
    zero_cov = torch.zeros(2, 3, 3, dtype=torch.complex64)
    zero_coh = coherenceMatrix(zero_cov)
    eye = torch.eye(3, dtype=torch.complex64).expand(2, 3, 3)
    assert torch.isfinite(zero_coh).all()
    assert torch.allclose(zero_coh, eye, atol=1e-6)
    assert torch.allclose(gmsc(zero_coh), torch.zeros(2, 1, 1), atol=1e-6)

    # 3. Zero-padded channel (channel 1 muted)
    padded_cov = torch.tensor(
        [
            [4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 2.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 - 2.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    padded_coh = coherenceMatrix(padded_cov)
    assert torch.isfinite(padded_coh).all()
    assert torch.allclose(padded_coh[0, 2], torch.tensor(0.0 + 1.0j), atol=1e-6)
    assert torch.allclose(padded_coh[1, :], torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex64), atol=1e-6)

    # 4. Tiny negative real/imag drift + overshoot slightly > 1.0 + M=1 edge case
    drift_cov = torch.tensor(
        [
            [1.0 - 1e-7j, 1.00001 + 0.0j],
            [1.00001 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    drift_coh = coherenceMatrix(drift_cov)
    assert torch.isfinite(drift_coh).all()
    assert (drift_coh.abs() <= 1.0 + 1e-6).all()

    neg_diag = torch.tensor([[-1e-18 + 1e-19j]], dtype=torch.complex64)
    neg_coh = coherenceMatrix(neg_diag)
    assert torch.allclose(neg_coh, torch.ones((1, 1), dtype=torch.complex64))


def test_pure_stft_polar_and_cartesian_modes() -> None:
    """Verifies PureSTFTFeatureExtractor polar and cartesian formats in ref and all modes."""
    transform = _make_stft_transform()
    F = transform.num_freq_bins
    B, M, T = 2, 4, 8

    stft = torch.randn(B, M, F, T, dtype=torch.complex64)
    # Inject an all-zero frame to confirm silent zero-safety
    stft[:, :, :, 0] = 0.0 + 0.0j

    # Polar + all mode
    ext_polar_all = PureSTFTFeatureExtractor(transform, representation="polar", mode="all")
    assert ext_polar_all.feature_dim == 3 * F
    out_pa = ext_polar_all.forward_stft(stft)
    assert out_pa.shape == (B, M, 3 * F, T)
    assert torch.isfinite(out_pa).all()
    assert torch.allclose(out_pa[:, :, :, 0], torch.zeros(B, M, 3 * F))

    mag = out_pa[:, :, :F, 1:]
    cos_phi = out_pa[:, :, F : 2 * F, 1:]
    sin_phi = out_pa[:, :, 2 * F :, 1:]
    expected_mag = torch.abs(stft[:, :, :, 1:])
    assert torch.allclose(mag, expected_mag, atol=1e-6)
    assert torch.allclose(cos_phi.pow(2) + sin_phi.pow(2), torch.ones_like(mag), atol=1e-5)

    valid_mics = torch.tensor([2, 4])
    mask_all = ext_polar_all.get_valid_feature_mask(valid_mics, max_M=M)
    assert mask_all is not None
    assert torch.equal(
        mask_all,
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )

    # Polar + ref mode (ref_channel=2)
    ext_polar_ref = PureSTFTFeatureExtractor(
        transform, representation="polar", mode="ref", ref_channel=2
    )
    out_pr = ext_polar_ref.forward_stft(stft)
    assert out_pr.shape == (B, 1, 3 * F, T)
    assert torch.allclose(out_pr[:, 0], out_pa[:, 2], atol=1e-6)
    mask_ref = ext_polar_ref.get_valid_feature_mask(valid_mics, max_M=M)
    assert mask_ref is not None
    assert torch.equal(mask_ref, torch.tensor([[True], [True]]))

    # Backward-compatible Cartesian + all / ref modes
    ext_cart_all = PureSTFTFeatureExtractor(transform, representation="cartesian", mode="all")
    assert ext_cart_all.feature_dim == 2 * F
    out_ca = ext_cart_all.forward_stft(stft)
    assert out_ca.shape == (B, M, 2 * F, T)
    assert torch.allclose(out_ca[:, :, :F], stft.real)
    assert torch.allclose(out_ca[:, :, F:], stft.imag)

    ext_cart_ref = PureSTFTFeatureExtractor(
        transform, representation="cartesian", mode="ref", ref_channel=1
    )
    out_cr = ext_cart_ref.forward_stft(stft)
    assert out_cr.shape == (B, 1, 2 * F, T)
    assert torch.allclose(out_cr[:, 0], out_ca[:, 1])

    # Unbatched 3D STFT input via forward(..., input_type="stft") with shape (F, M, T)
    unbatched = stft[0].transpose(0, 1)  # (F, M, T)
    out_unbatched = ext_polar_ref(unbatched, input_type="stft")
    assert out_unbatched.shape == (1, 3 * F, T)


def test_psc_matches_reference_coherence_and_handles_edge_cases() -> None:
    """Verifies PSC_Feature_Extractor against full matrix coherence, causality, and masks."""
    torch.manual_seed(42)
    transform = _make_stft_transform()
    F = transform.num_freq_bins
    B, M, T = 2, 4, 15
    P_all = M * (M - 1) // 2

    stft = torch.randn(B, M, F, T, dtype=torch.complex64)
    psc_all = PSC_Feature_Extractor(
        transform, smoothing_time_constant=0.1, mode="all"
    )
    assert psc_all.feature_dim == 3 * F
    assert not psc_all.is_trainable
    out_all = psc_all.forward_stft(stft)
    assert out_all.shape == (B, P_all, 3 * F, T)
    assert torch.isfinite(out_all).all()

    # Compare against explicit full covariance matrix + coherenceMatrix reference
    stft_d = stft.to(torch.complex128)
    # Full SCM outer product: (B, F, T, M, M)
    y = stft_d.permute(0, 2, 3, 1)  # (B, F, T, M)
    scm = y.unsqueeze(-1) @ y.conj().unsqueeze(-2)
    alpha = transform.timeConstant2smoothingFactor(0.1)
    smooth_cov = exp_windowing_recursive(scm, smoothing_factor=alpha, dim=-3)
    coh_mat = coherenceMatrix(smooth_cov)  # (B, F, T, M, M)
    row, col = torch.triu_indices(M, M, offset=1)
    # Extract (B, P, F, T) complex
    gamma_ref = coh_mat[..., row, col].permute(0, 3, 1, 2)

    mag_psc = out_all[:, :, :F, :]
    cos_psc = out_all[:, :, F : 2 * F, :]
    sin_psc = out_all[:, :, 2 * F :, :]
    expected_mag = gamma_ref.abs().clamp(max=1.0).to(torch.float32)
    assert torch.allclose(mag_psc, expected_mag, atol=1e-5)
    assert torch.allclose(
        cos_psc, (gamma_ref.real / gamma_ref.abs()).to(torch.float32), atol=1e-5
    )
    assert torch.allclose(
        sin_psc, (gamma_ref.imag / gamma_ref.abs()).to(torch.float32), atol=1e-5
    )
    assert (mag_psc >= 0.0).all() and (mag_psc <= 1.0 + 1e-6).all()

    # Check mode="ref" matches subset paired against ref_channel=1
    psc_ref = PSC_Feature_Extractor(
        transform, smoothing_time_constant=0.1, mode="ref", ref_channel=1
    )
    out_ref = psc_ref.forward_stft(stft)
    assert out_ref.shape == (B, M - 1, 3 * F, T)
    # Also verify exact match to coherenceMatrix[:, :, :, others, 1]
    others = torch.tensor([0, 2, 3])
    gamma_ref1 = coh_mat[:, :, :, others, 1].permute(0, 3, 1, 2)
    assert torch.allclose(out_ref[:, :, :F, :], gamma_ref1.abs().to(torch.float32), atol=1e-5)

    # Check causal invariance: perturbing frames t >= 7 cannot affect frames t < 7
    stft_perturbed = stft.clone()
    stft_perturbed[:, :, :, 7:] *= 10.0
    out_perturbed = psc_all.forward_stft(stft_perturbed)
    assert torch.allclose(out_all[:, :, :, :7], out_perturbed[:, :, :, :7], atol=1e-6)

    # Check instantaneous mode (smoothing_time_constant=0.0): |Gamma| == 1 and phase matches block-concatenated CSIPD
    psc_inst = PSC_Feature_Extractor(
        transform, smoothing_time_constant=0.0, mode="all"
    )
    csipd_all = CSIPD_Feature_Extractor(transform, mode="all")
    out_inst = psc_inst.forward_stft(stft)
    out_csipd = csipd_all.forward_stft(stft)
    assert torch.allclose(out_inst[:, :, :F, :], torch.ones(B, P_all, F, T), atol=1e-6)
    assert torch.allclose(out_inst[:, :, F : 2 * F, :], out_csipd[:, :, :F, :], atol=1e-5)
    assert torch.allclose(out_inst[:, :, 2 * F :, :], out_csipd[:, :, F : 2 * F, :], atol=1e-5)

    # Check silent channels + valid_feature_mask
    stft_muted = stft.clone()
    stft_muted[:, 2:, :, :] = 0.0
    out_muted = psc_all.forward_stft(stft_muted)
    assert torch.isfinite(out_muted).all()
    valid_mics = torch.tensor([2, 3])
    mask_all = psc_all.get_valid_feature_mask(valid_mics, max_M=M)
    # Triu pairs for M=4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    assert mask_all is not None
    assert torch.equal(
        mask_all,
        torch.tensor(
            [
                [True, False, False, False, False, False],
                [True, True, False, True, False, False],
            ]
        ),
    )

    # Single-mic edge case M=1 -> P=0 without error
    out_single = psc_all.forward_stft(stft[:, :1])
    assert out_single.shape == (B, 0, 3 * F, T)
    mask_single = psc_all.get_valid_feature_mask(torch.tensor([1, 1]), max_M=1)
    assert mask_single is not None and mask_single.shape == (2, 0)


def test_hydra_configs_and_stacked_integration() -> None:
    """Verifies Hydra instantiation of pure_stft.yaml, psc.yaml, and StackedFeatureExtractor."""
    transform = _make_stft_transform()
    config_dir = "/data4/Henri/MuSE-Toolbox/configs/model/feature_extractor"
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_pure = compose(config_name="pure_stft")
        partial_pure = instantiate(cfg_pure)
        assert isinstance(partial_pure, functools.partial)
        fe_pure = partial_pure(transform=transform)
        assert isinstance(fe_pure, PureSTFTFeatureExtractor)
        assert fe_pure.representation == "polar"
        assert fe_pure.mode == "all"

        cfg_psc = compose(config_name="psc")
        partial_psc = instantiate(cfg_psc)
        assert isinstance(partial_psc, functools.partial)
        fe_psc = partial_psc(transform=transform)
        assert isinstance(fe_psc, PSC_Feature_Extractor)
        assert fe_psc.smoothing_time_constant == 0.1
        assert fe_psc.mode == "all"

    stacked = StackedFeatureExtractor(
        transform=transform,
        extractors=[
            {"PureSTFTFeatureExtractor": {"representation": "polar", "mode": "all"}},
            {"PSC_Feature_Extractor": {"smoothing_time_constant": 0.1, "mode": "all"}},
        ],
    )
    assert stacked.feature_dim == 6 * transform.num_freq_bins
