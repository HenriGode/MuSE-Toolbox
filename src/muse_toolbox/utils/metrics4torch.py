import torch

# from pesq import pesq_batch, PesqError
# from pystoi import stoi
import concurrent.futures
from multiprocessing import Pool, Queue, Process, cpu_count
from functools import partial
import numpy as np


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

    W = torch.tensor(
        [
            0.003,
            0.003,
            0.003,
            0.007,
            0.010,
            0.016,
            0.016,
            0.017,
            0.017,
            0.022,
            0.027,
            0.028,
            0.030,
            0.032,
            0.034,
            0.035,
            0.037,
            0.036,
            0.036,
            0.033,
            0.030,
            0.029,
            0.027,
            0.026,
            0.026,
        ],
        device=ref.device,
        dtype=ref.dtype,
    )

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


# def pesq_wrapper(
#     ref: torch.Tensor,
#     sig: torch.Tensor,
#     fs: float,
#     mode: str = "wb",
#     num_workers: int = 1,
#     on_error=PesqError.RAISE_EXCEPTION,
# ) -> torch.Tensor:

#     # Broadcast ref and sig to a common shape
#     ref, sig = torch.broadcast_tensors(ref, sig)
#     # Flatten all dimensions except the last (time) dimension
#     flat_shape = (-1, ref.shape[-1])
#     ref_flat = ref.reshape(flat_shape)
#     sig_flat = sig.reshape(flat_shape)

#     # Convert to numpy arrays
#     ref_np = ref_flat.cpu().numpy()
#     sig_np = sig_flat.cpu().numpy()

#     vals = pesq_batch(
#         fs, ref_np, sig_np, mode=mode, n_processor=num_workers, on_error=on_error
#     )

#     # Reshape results to broadcast shape (excluding time dimension)
#     out_shape = ref.shape[:-1]
#     result = torch.tensor(vals, dtype=ref.dtype, device=ref.device).reshape(out_shape)[
#         ..., None
#     ]
#     print(result.shape)
#     return result


# def stoi_wrapper(
#     ref: torch.Tensor,
#     sig: torch.Tensor,
#     fs: int,
#     extended: bool = False,
#     num_workers: int = 1,
# ) -> torch.Tensor:
#     # Broadcast ref and sig to a common shape
#     ref, sig = torch.broadcast_tensors(ref, sig)
#     # Flatten all dimensions except the last (time) dimension
#     flat_shape = (-1, ref.shape[-1])
#     ref_flat = ref.reshape(flat_shape)
#     sig_flat = sig.reshape(flat_shape)
#     # Convert to numpy arrays
#     ref_np = ref_flat.cpu().numpy()
#     sig_np = sig_flat.cpu().numpy()

#     vals = stoi_batch(
#         int(fs), ref_np, sig_np, extended=extended, n_processor=num_workers
#     )

#     # Reshape results to broadcast shape (excluding time dimension)
#     out_shape = ref.shape[:-1]
#     result = torch.tensor(vals, dtype=ref.dtype, device=ref.device).reshape(out_shape)[
#         ..., None
#     ]
#     print(result.shape)
#     return result


def _processor_coordinator(func, args_q, results_q):
    while True:
        index, arg = args_q.get()
        if index is None:
            break
        try:
            result = func(*arg)
        except Exception as e:
            result = e
        results_q.put((index, result))


def _processor_mapping(func, args, n_processor):
    args_q = Queue(maxsize=1)
    results_q = Queue()
    processors = [
        Process(target=_processor_coordinator, args=(func, args_q, results_q))
        for _ in range(n_processor)
    ]
    for p in processors:
        p.daemon = True
        p.start()
    for i, arg in enumerate(args):
        args_q.put((i, arg))
    # send stop messages
    for _ in range(n_processor):
        args_q.put((None, None))
    results = [results_q.get() for _ in range(len(args))]
    [p.join() for p in processors]
    return [v[1] for v in sorted(results)]


# def stoi_batch(
#     fs: int,
#     ref: np.ndarray,
#     deg: np.ndarray,
#     extended: bool = False,
#     n_processor: int = cpu_count(),
# ):
#     """
#     Running `stoi` using multiple processors
#     Args:
#         on_error:
#         ref: numpy 1D (n_sample,) or 2D array (n_file, n_sample), reference audio signal
#         deg: numpy 1D (n_sample,) or 2D array (n_file, n_sample), degraded audio signal
#         fs:  integer, sampling rate
#         extended: False (default) or True (extended STOI)
#         n_processor: cpu_count() (default) or number of processors (chosen by the user) or 0 (without multiprocessing)
#     Returns:
#         stoi_score: list of stoi scores
#     """
#     # check dimension
#     if len(ref.shape) == 1:
#         if len(deg.shape) == 1 and ref.shape == deg.shape:
#             return [stoi(ref, deg, fs, extended)]
#         elif len(deg.shape) == 2 and ref.shape[-1] == deg.shape[-1]:
#             if n_processor <= 0:
#                 stoi_score = [np.nan for i in range(deg.shape[0])]
#                 for i in range(deg.shape[0]):
#                     stoi_score[i] = stoi(ref, deg[i, :], fs, extended)
#                 return stoi_score
#             else:
#                 with Pool(n_processor) as p:
#                     return p.map(
#                         partial(stoi, ref, fs_sig=fs, extended=extended),
#                         [deg[i, :] for i in range(deg.shape[0])],
#                     )
#         else:
#             raise ValueError("The shapes of `deg` is invalid!")
#     elif len(ref.shape) == 2:
#         if deg.shape == ref.shape:
#             if n_processor <= 0:
#                 stoi_score = [np.nan for i in range(deg.shape[0])]
#                 for i in range(deg.shape[0]):
#                     stoi_score[i] = stoi(ref[i, :], deg[i, :], fs, extended)
#                 return stoi_score
#             else:
#                 return _processor_mapping(
#                     stoi,
#                     [(ref[i, :], deg[i, :], fs, extended) for i in range(deg.shape[0])],
#                     n_processor,
#                 )
#         else:
#             raise ValueError("The shape of `deg` is invalid!")
#     else:
#         raise ValueError("The shape of `ref` should be either 1D or 2D!")


# def stoi_wrapper_for_loop(
#     ref: torch.Tensor,
#     sig: torch.Tensor,
#     fs: float,
#     extended: bool = False,
# ) -> torch.Tensor:

#     # Broadcast ref and sig to a common shape
#     ref, sig = torch.broadcast_tensors(ref, sig)
#     # Flatten all dimensions except the last (time) dimension
#     flat_shape = (-1, ref.shape[-1])
#     ref_flat = ref.reshape(flat_shape)
#     sig_flat = sig.reshape(flat_shape)
#     # Convert to numpy arrays
#     ref_np = ref_flat.cpu().numpy()
#     sig_np = sig_flat.cpu().numpy()
#     # Compute STOI
#     results = []
#     for i in range(ref_np.shape[0]):
#         val = stoi(ref_np[i], sig_np[i], fs, extended=extended)
#         results.append(val)
#     result = torch.tensor(results, dtype=ref.dtype, device=ref.device).reshape(
#         ref.shape[:-1]
#     )[..., None]
#     return result


# def _stoi_worker(args):
#     ref_arr, sig_arr, fs, extended = args
#     return stoi(ref_arr, sig_arr, fs, extended=extended)


# def stoi_wrapper_multi(
#     ref: torch.Tensor,
#     sig: torch.Tensor,
#     fs: float,
#     extended: bool = False,
#     num_workers: int = 1,
# ) -> torch.Tensor:
#     # Broadcast ref and sig to a common shape
#     ref, sig = torch.broadcast_tensors(ref, sig)
#     # Flatten all dimensions except the last (time) dimension
#     flat_shape = (-1, ref.shape[-1])
#     ref_flat = ref.reshape(flat_shape)
#     sig_flat = sig.reshape(flat_shape)
#     # Convert to numpy arrays
#     ref_np = ref_flat.cpu().numpy()
#     sig_np = sig_flat.cpu().numpy()

#     if num_workers > 1:
#         args_iter = (
#             (ref_np[i], sig_np[i], int(fs), extended) for i in range(ref_np.shape[0])
#         )
#         with concurrent.futures.ProcessPoolExecutor(
#             max_workers=num_workers
#         ) as executor:
#             results = list(executor.map(_stoi_worker, args_iter))
#     else:
#         results = [
#             _stoi_worker((ref_np[i], sig_np[i], int(fs), extended))
#             for i in range(ref_np.shape[0])
#         ]

#     result = torch.tensor(results, dtype=ref.dtype, device=ref.device).reshape(
#         ref.shape[:-1]
#     )[..., None]
#     return result


def compute_ref_based_metrics(refs, sigs, fs, metrics, **kwargs):
    t = 5
    return None
