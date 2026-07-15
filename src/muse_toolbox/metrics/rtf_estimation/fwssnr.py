import torch
from muse_toolbox.metrics.rtf_estimation.ref_metric import RefMetric
from muse_toolbox.utils import STFTtransform


def fwssnr(
    ref: torch.Tensor, sig: torch.Tensor, fs: float, frameLen=0.03, overlap=0.75
) -> torch.Tensor:

    winlength = round(frameLen * fs)
    skiprate = int(
        torch.floor(torch.tensor((1 - overlap) * frameLen * fs))
    )  # window skip in samples
    max_freq = fs / 2  # maximum bandwidth
    num_crit = 25  # number of critical bands
    n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(2 * winlength))))
    n_fftby2 = int(n_fft / 2)
    gamma = 0.2

    cent_freq = torch.zeros((num_crit,), device=ref.device, dtype=ref.dtype)
    bandwidth = torch.zeros((num_crit,), device=ref.device, dtype=ref.dtype)

    cent_freq[0] = 50.0000
    bandwidth[0] = 70.0000
    cent_freq[1] = 120.000
    bandwidth[1] = 70.0000
    cent_freq[2] = 190.000
    bandwidth[2] = 70.0000
    cent_freq[3] = 260.000
    bandwidth[3] = 70.0000
    cent_freq[4] = 330.000
    bandwidth[4] = 70.0000
    cent_freq[5] = 400.000
    bandwidth[5] = 70.0000
    cent_freq[6] = 470.000
    bandwidth[6] = 70.0000
    cent_freq[7] = 540.000
    bandwidth[7] = 77.3724
    cent_freq[8] = 617.372
    bandwidth[8] = 86.0056
    cent_freq[9] = 703.378
    bandwidth[9] = 95.3398
    cent_freq[10] = 798.717
    bandwidth[10] = 105.411
    cent_freq[11] = 904.128
    bandwidth[11] = 116.256
    cent_freq[12] = 1020.38
    bandwidth[12] = 127.914
    cent_freq[13] = 1148.30
    bandwidth[13] = 140.423
    cent_freq[14] = 1288.72
    bandwidth[14] = 153.823
    cent_freq[15] = 1442.54
    bandwidth[15] = 168.154
    cent_freq[16] = 1610.70
    bandwidth[16] = 183.457
    cent_freq[17] = 1794.16
    bandwidth[17] = 199.776
    cent_freq[18] = 1993.93
    bandwidth[18] = 217.153
    cent_freq[19] = 2211.08
    bandwidth[19] = 235.631
    cent_freq[20] = 2446.71
    bandwidth[20] = 255.255
    cent_freq[21] = 2701.97
    bandwidth[21] = 276.072
    cent_freq[22] = 2978.04
    bandwidth[22] = 298.126
    cent_freq[23] = 3276.17
    bandwidth[23] = 321.465
    cent_freq[24] = 3597.63
    bandwidth[24] = 346.136

    

    bw_min = bandwidth[0]
    min_factor = torch.exp(
        torch.tensor(-30.0 / (2.0 * 2.303))
    )  #      % -30 dB point of filter

    all_f0 = torch.zeros((num_crit,), device=ref.device, dtype=ref.dtype)
    crit_filter = torch.zeros(
        (num_crit, int(n_fftby2)), device=ref.device, dtype=ref.dtype
    )
    j = torch.arange(0, n_fftby2, device=ref.device, dtype=ref.dtype)

    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = torch.floor(f0)
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = torch.log(bw_min) - torch.log(bandwidth[i])
        crit_filter[i, :] = torch.exp(
            -11 * (((j - torch.floor(f0)) / bw) ** 2) + norm_factor
        )
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    num_frames = ref.shape[-1] / skiprate - (winlength / skiprate)  # number of frames

    hannWin = 0.5 * (
        1
        - torch.cos(
            2
            * torch.pi
            * torch.arange(1, winlength + 1, device=ref.device, dtype=ref.dtype)
            / (winlength + 1)
        )
    )
    ref_reshaped = ref.reshape(-1, ref.shape[-1])
    zeros = torch.zeros(
        (ref_reshaped.shape[0], int((n_fft - winlength) / 2)), device=ref.device
    )
    ref_tmp = torch.cat(
        [
            zeros,
            ref_reshaped[:, 0 : int(num_frames) * skiprate + int(winlength - skiprate)],
            zeros,
        ],
        dim=-1,
    )
    Zxx = torch.stft(
        ref_tmp,
        n_fft=n_fft,
        hop_length=skiprate,
        win_length=hannWin.shape[-1],
        window=hannWin,
        center=False,
        onesided=True,
        return_complex=True,
    )
    ref_spec = torch.abs(Zxx).reshape(*ref.shape[:-1], *Zxx.shape[-2:])
    ref_spec = ref_spec[..., :-1, :]
    ref_spec = ref_spec / ref_spec.sum(dim=-2, keepdim=True)
    sig_reshaped = sig.reshape(-1, sig.shape[-1])
    zeros = torch.zeros(
        (sig_reshaped.shape[0], int((n_fft - winlength) / 2)), device=ref.device
    )
    sig_tmp = torch.cat(
        [
            zeros,
            sig_reshaped[:, 0 : int(num_frames) * skiprate + int(winlength - skiprate)],
            zeros,
        ],
        dim=-1,
    )
    Zxx = torch.stft(
        sig_tmp,
        n_fft=n_fft,
        hop_length=skiprate,
        win_length=hannWin.shape[-1],
        window=hannWin,
        center=False,
        onesided=True,
        return_complex=True,
    )
    sig_spec = torch.abs(Zxx).reshape(*sig.shape[:-1], *Zxx.shape[-2:])
    sig_spec = sig_spec[..., :-1, :]
    sig_spec = sig_spec / sig_spec.sum(dim=-2, keepdim=True)

    ref_energy = crit_filter @ ref_spec
    sig_energy = crit_filter @ sig_spec
    error_energy = torch.pow(ref_energy - sig_energy, 2)
    error_energy[error_energy < torch.finfo(torch.float32).eps ** 2] = (
        torch.finfo(torch.float32).eps ** 2
    )
    W_freq = torch.pow(ref_energy, gamma)
    SNRlog = 10 * torch.log10((ref_energy**2) / error_energy)
    fwSNR = torch.sum(W_freq * SNRlog, dim=-2, keepdim=True) / torch.sum(
        W_freq, dim=-2, keepdim=True
    )
    distortion = fwSNR.clone()
    distortion[distortion < -10] = -10
    distortion[distortion > 35] = 35

    return torch.mean(distortion, dim=-1, keepdim=False)


class FWSSNR(RefMetric):
    """Frequency-Weighted Segmental SNR (FWSSNR) metric class.
    
    Inherits from `RefMetric` to evaluate the Frequency-Weighted Segmental SNR 
    between degraded and reference audio signals.
    """
    def __init__(
        self,
        transform: STFTtransform,
        frameLen: float,
        overlap: float,
        ref_channels: list[int],
        *args,
        **kwargs
    ):
        """Initializes the FWSSNR metric.

        Args:
            transform (STFTtransform): Transformer object for STFT handling.
            frameLen (float): Frame length in seconds.
            overlap (float): Overlap ratio between frames (e.g., 0.75).
            ref_channels (list[int]): Reference channels to evaluate.
            *args: Additional arguments passed to RefMetric.
            **kwargs: Additional keyword arguments passed to RefMetric.
        """
        super().__init__(
            metric_name="FWSSNR",
            transform=transform,
            ref_channels=ref_channels,
            *args,
            **kwargs
        )

        self.frameLen = frameLen
        self.overlap = overlap
        self.fs = int(self.transform.sampling_frequency)

    def evaluate_metric(self, deg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Evaluates the FWSSNR for a given degraded and reference signal pair.

        Args:
            deg (torch.Tensor): The degraded (predicted) signal tensor.
            ref (torch.Tensor): The reference (ground truth) signal tensor.

        Returns:
            torch.Tensor: Evaluated FWSSNR score.
        """
        try:
            return fwssnr(
                ref=ref.squeeze(0),
                sig=deg.squeeze(0),
                fs=self.fs,
                frameLen=self.frameLen,
                overlap=self.overlap,
            )
        except Exception:
            return torch.tensor([float("nan")], device=ref.device)
