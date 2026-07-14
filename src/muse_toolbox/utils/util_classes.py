# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import random
import os.path as path
import natsort
import glob
import itertools as it
import torch

from .dsp.acoustic_simulation import convolve_clean2microphone, convolve_white2microphone
from .dsp.stats import coherenceMatrix, gmsc, smoothCovarianceMatrix
from .dsp.transforms import STFTtransform
from .dsp.whitening import noise_subtraction, noise_whitening
from .math.complex_angles import hermitian_angle
from .math.conversions import db2amp, pow2db
from .math.covariance import covariance_SCM, make2covariance_matrix
from .math.geometry import slerp
from .math.matrix_ops import characteristic_subspace, characteristic_subspace_h, makeHermitian, makeMatrixUnitNorm, makeVectorUnitNorm, oblique_projection, orthogonal_complement, orthogonal_projection, parallel_projection, peigvech, regularize, trace
from .math.stochastics import randdir, wmean
from .math.windowing import exp_windowing, windowing
from .tensor_ops import *
from .system import *
from .metrics4torch import *
import pickle
from typing import Union, Tuple, List

# %%


class Clean_speech:

    def __init__(
        self,
        filename: str,
        start_time: float,
        signal_len: float,
        lin_onset_len=50e-3,
        device: str = "cuda:0",
        dtype=torch.float64,
    ) -> None:
        self.filename = filename  # 'f000_000.wav'
        self.speaker_id = path.basename(filename)[0:6]  # 'f00000'
        self.sex = path.basename(filename)[0]  # 'm', 'f'
        self.data, self.sr = load_audio(filename, device=device, dtype=dtype)
        self.data = self.data[0, : int(signal_len * self.sr)] * torch.cat(
            [
                torch.linspace(0, 1, int(lin_onset_len * self.sr), device=device),
                torch.ones(
                    int(signal_len * self.sr) - int(lin_onset_len * self.sr),
                    device=device,
                ),
            ],
            dim=-1,
        )
        if start_time is not None:
            self.data = torch.cat(
                [
                    torch.zeros(
                        int(start_time * self.sr),
                        device=self.data.device,
                        dtype=self.data.dtype,
                    ),
                    self.data[: self.data.shape[-1] - int(start_time * self.sr)],
                ]
            )


class Clean_speech_list:
    def __init__(
        self,
        file_path,
        num_speakers_per_sex=3,
        num_utterances_per_speaker=2,
        sexes="mf",
    ) -> None:
        self.file_path = file_path
        self.num_speakers_per_sex = num_speakers_per_sex
        self.num_utterances_per_speaker = num_utterances_per_speaker
        self.sexes = sexes
        self.allfiles = natsort.natsorted(glob.glob(path.join(file_path, "*.wav")))
        self.files2use = list(
            it.compress(
                self.allfiles,
                [
                    path.basename(fn)[0] in sexes
                    and path.basename(fn)[0:6]
                    in ["f00061", "f00005", "f00007", "m00083", "m00023", "m00054"]
                    # and int(path.basename(fn)[1:6 ]) < num_speakers_per_sex
                    # and int(path.basename(fn)[7:10]) > 4
                    # and int(path.basename(fn)[7:10]) < 5+num_utterances_per_speaker
                    for fn in self.allfiles
                ],
            )
        )

    def all_permutations(self, num_speakers=2):
        return list(it.permutations(self.files2use, num_speakers))


class RIR:
    def __init__(
        self, filename: str, device: str = "cuda:0", dtype=torch.float64
    ) -> None:
        self.filename = filename
        self.room = path.basename(filename).split("_")[1]
        self.array = path.basename(filename).split("_")[3][:-4]
        self.source_location = path.basename(filename).split("_")[2]
        self.data, self.sr = load_audio(filename, device=device, dtype=dtype)

    def direct(
        self, time_after_max_power=50e-3, provide_tail: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        max_power_sample = torch.linalg.vector_norm(
            self.data, ord=2, dim=(-2), keepdim=True
        ).max(dim=-1)[1]
        direct = self.data[
            :, : (max_power_sample + int(time_after_max_power * self.sr))
        ]
        tail = (
            self.data[:, (max_power_sample + int(time_after_max_power * self.sr)) :]
            if provide_tail
            else None
        )
        return direct, tail

    def oracleRTF(
        self, transform=STFTtransform(), time_after_max_power=50e-3
    ) -> (
        torch.Tensor
    ):  # TODO: store these in signal object to not compute them multiple times !!!
        direct = self.data  # self.direct(time_after_max_power=time_after_max_power)[0]
        directwhite = convolve_white2microphone(direct, samples=1600000)
        stftsig = transform.encode(directwhite)
        covMat = covariance_SCM(stftsig)[..., None, :, :]
        # covMat_T    = covariance_Tyler(stftsig)[...,None,:,:]
        return peigvech(covMat)


class RIR_list:

    def __init__(
        self, file_path: str, rooms=["0310ms"], microphone_arrays=["BTE2x2center"]
    ) -> None:
        self.file_path = file_path
        self.allfiles = natsort.natsorted(glob.glob(path.join(file_path, "*.wav")))
        self.rooms = set(
            [path.basename(fn).split("_")[1] for fn in self.allfiles]
        ).intersection(set(rooms))
        self.microphone_arrays = set(
            [path.basename(fn).split("_")[3][:-4] for fn in self.allfiles]
        ).intersection(set(microphone_arrays))
        self.source_positions = set(
            [path.basename(fn).split("_")[2] for fn in self.allfiles]
        )

    def all_combinations(self, num_sources=2, min_angle=30):
        all_source_pos_combis = list(
            it.permutations(self.source_positions, num_sources)
        )
        if num_sources > 1:
            source_pos_combis_2use = [
                spc
                for spc in all_source_pos_combis
                if min(
                    [self._angular_distance(ac) for ac in list(it.combinations(spc, 2))]
                )
                >= min_angle
            ]
        else:
            source_pos_combis_2use = all_source_pos_combis
        all_combis = list(
            it.product(self.rooms, self.microphone_arrays, source_pos_combis_2use)
        )
        return [
            tuple(
                [
                    path.join(
                        self.file_path,
                        "RIR_" + combi[0] + "_" + angle + "_" + combi[1] + ".wav",
                    )
                    for angle in combi[2]
                ]
            )
            for combi in all_combis
        ]

    def _angular_distance(self, angle_pair):
        return 180 - abs(
            abs(int(angle_pair[0][1:]) - int(angle_pair[1][1:])) % 360 - 180
        )


class Noise:

    def __init__(
        self,
        filename: str,
        signal_len: float,
        device: str = "cuda:0",
        dtype=torch.float64,
    ) -> None:
        self.filename = filename
        self.room = path.basename(filename).split("_")[1]
        self.array = path.basename(filename).split("_")[2]
        self.noise_type = path.basename(filename).split("_")[3][:-4]
        self.data, self.sr = load_audio(filename, device=device, dtype=dtype)
        self.data = self.data[:, : int(signal_len * self.sr)]


class Noise_list:

    def __init__(
        self,
        file_path: str,
        rooms=["0520ms"],
        microphone_arrays=["BTE2x2center"],
        noise_types=["white"],
    ) -> None:
        self.file_path = file_path
        self.allfiles = natsort.natsorted(glob.glob(path.join(file_path, "*.wav")))
        self.rooms = set(
            [path.basename(fn).split("_")[1] for fn in self.allfiles]
        ).intersection(set(rooms))
        self.microphone_arrays = set(
            [path.basename(fn).split("_")[2] for fn in self.allfiles]
        ).intersection(set(microphone_arrays))
        self.noise_types = set(
            [path.basename(fn).split("_")[3][:-4] for fn in self.allfiles]
        ).intersection(set(noise_types))


class Signal_list:

    def __init__(
        self,
        clean_speeches: Clean_speech_list,
        RIRs: RIR_list,
        noises: Noise_list,
        num_sources=2,
        min_angle=30,
    ) -> None:
        self.clean_speeches = clean_speeches
        self.RIRs = RIRs
        self.noises = noises
        self.num_sources = num_sources
        self.min_angle = min_angle
        self.all = self.all_combinations()
        random.shuffle(self.all)

    def all_combinations(self):
        # combis2 = list(it.product(self.clean_speeches.all_permutations(self.num_sources),
        #                          self.RIRs.all_combinations(self.num_sources, self.min_angle),
        #                          self.noises.noise_types))
        tmp = list(
            it.product(
                self.RIRs.all_combinations(self.num_sources, self.min_angle),
                self.noises.noise_types,
            )
        )
        clean_speech = self.clean_speeches.all_permutations(self.num_sources)
        combis = [
            (clean_speech[np.mod(idx, len(clean_speech))],) + rirnoise
            for idx, rirnoise in enumerate(tmp)
        ]
        return [
            (
                combi[0],
                combi[1],
                path.join(
                    self.noises.file_path,
                    "Noise_"
                    + path.basename(combi[1][0]).split("_")[1]
                    + "_"
                    + path.basename(combi[1][0]).split("_")[3][:-4]
                    + "_"
                    + combi[2]
                    + ".wav",
                ),
            )
            for combi in combis
        ]

    def speaker_count(
        self,
        Ry: torch.Tensor,
        only_noise_frames: int,
        threshold: float,
        freq_weighting: torch.Tensor,
        SNR: int,
    ) -> torch.Tensor:
        spk_start = [torch.tensor(0, device=Ry.device)]
        time_constant = 1.0
        transform = STFTtransform(frame_length=64e-3, frame_shift=16e-3)
        smoothing_factor = transform.timeConstant2smoothingFactor(time_constant)
        Ry = Ry[..., 1:-1, :, :, :]

        calcWMSC = False
        if not calcWMSC:
            if threshold is None:
                with open(
                    "/data2/Henri/journal2_CB/Code/SC_thresholds64.pkl", "rb"
                ) as file:
                    thresholds = pickle.load(file)
                snridx = int(-SNR / 5)
        else:
            spk_start = torch.tensor([0, 21, 121, 221, 321])
            spk_start = torch.tensor([0, 251, 501, 751, 1001])
            gamma_all = torch.tensor([]).cuda()

        for spk in [1, 2, 3]:
            Rn = Ry[..., [only_noise_frames + spk_start[spk - 1]], :, :]
            Rz = noise_whitening(Rn, Ry)[1]
            gammac_freqs = gmsc(coherenceMatrix(Rz))
            gammac = wmean(gammac_freqs, dims=-4, weights=freq_weighting).squeeze()
            gamman_freqs = gmsc(
                coherenceMatrix(Rz + torch.eye(Rz.shape[-1], device=Rz.device))
            )
            gamman = wmean(gamman_freqs, dims=-4, weights=freq_weighting).squeeze()
            gamma_freqs = gamman_freqs / gammac_freqs
            gamma = (gamman / gammac).squeeze()
            gamma[0 : only_noise_frames + spk_start[spk - 1] + 1] = 0
            gamma = exp_windowing(data=gamma, smoothing_factor=smoothing_factor, dim=-1)

            if calcWMSC:
                # gamma  = gamman.squeeze()
                gamma2 = wmean(gamma_freqs, dims=-4, weights=freq_weighting).squeeze()
                gamman[0 : only_noise_frames + spk_start[spk - 1] + 1] = 0
                gammac[0 : only_noise_frames + spk_start[spk - 1] + 1] = 0
                gamma2[0 : only_noise_frames + spk_start[spk - 1] + 1] = 0

                gamman = exp_windowing(
                    data=gamman, smoothing_factor=smoothing_factor, dim=-1
                )
                gammac = exp_windowing(
                    data=gammac, smoothing_factor=smoothing_factor, dim=-1
                )
                gamma2 = exp_windowing(
                    data=gamma2, smoothing_factor=smoothing_factor, dim=-1
                )
            else:
                idx_vec = torch.linspace(
                    0,
                    Ry.shape[-3] - 1,
                    Ry.shape[-3],
                    device=gamma.device,
                    dtype=torch.int64,
                )[gamma.squeeze() > thresholds[1][snridx][spk - 1]]
                if len(idx_vec) > 0:
                    spk_start += [idx_vec[0]]
                    if spk_start[-1] > Ry.shape[-3] - only_noise_frames - 1:
                        break
                else:
                    break

        if calcWMSC:
            gamma_all = torch.cat(
                [gamma_all, torch.stack([gammac, gamman, gamma, gamma2])[:, None, :]],
                dim=-2,
            )
            return gamma_all
        else:
            spk_starts = torch.stack(
                spk_start
                + list(
                    torch.arange(Ry.shape[-3] - 3, Ry.shape[-3] + 1, device=Ry.device)[
                        len(spk_start) - 5 :
                    ]
                ),
                dim=0,
            )
            print(spk_starts)
            return spk_starts

    def speaker_count_new(
        self,
        signal: torch.Tensor,
        transform: STFTtransform = STFTtransform(
            frame_length=200e-3, frame_shift=50e-3
        ),
        time_constant: float = 1.0,
        threshold: float = 0.24,
    ) -> torch.Tensor:

        # spk_start = [torch.tensor(0, device=signal.device)]

        R = smoothCovarianceMatrix(
            signal,
            smoothing_factor=transform.timeConstant2smoothingFactor(time_constant),
        )
        R = regularize(make2covariance_matrix(R), reg_factor=1e-8)

        whiteidx = torch.arange(0, R.shape[-3], device=signal.device, dtype=torch.long)
        whiteidx = whiteidx - torch.min(
            whiteidx // 2, transform.times2frames(torch.tensor(1))
        )

        Rn0 = make2covariance_matrix(R[:, whiteidx, ...])
        Ln0, Rw0, _ = noise_whitening(make2covariance_matrix(Rn0), R)
        mix0 = torch.linalg.solve_triangular(
            Ln0.squeeze(), signal.transpose(-1, -2)[..., None], upper=False, left=True
        )[..., 0].transpose(-1, -2)

        C0 = coherenceMatrix(
            Rw0 + torch.eye(Rw0.shape[-1], device=signal.device, dtype=signal.dtype)
        )

        gamma0 = gmsc(C0)

        smoothmix0 = exp_windowing(
            mix0, transform.timeConstant2smoothingFactor(time_constant)
        )

        powerweights0 = abs(smoothmix0 ** (2)).mean(dim=1)[..., None, None]
        powerweights0 = powerweights0 / powerweights0.sum(dim=0, keepdim=True)

        gamma0wm = (
            (gamma0 * powerweights0).sum(dim=0, keepdim=True)
            / (powerweights0).sum(dim=0, keepdim=True)
        ).squeeze()

        # Compute moving average with a 1-second window
        raise NotImplementedError(
            "There is a potential error in the next lines of code! Check the variable window_size"
        )
        window_size = transform.times2frames(
            torch.tensor(1.0)
        )  # Convert 1 second to frames
        num_time_frames = gamma0wm.shape[-1]
        exp_window = torch.tensor(
            transform.timeConstant2smoothingFactor(1.0),
            dtype=get_real_dtype(gamma0wm),
            device=gamma0wm.device,
        ) ** (
            torch.arange(
                0,
                num_time_frames,
                dtype=get_real_dtype(gamma0wm),
                device=gamma0wm.device,
            )
        )
        ones_window = torch.ones(
            (window_size,), dtype=get_real_dtype(gamma0wm), device=gamma0wm.device
        )
        wgamma = windowing(gamma0wm, window=ones_window, dim=-1)
        # wgamma2 = windowing(gamma0wm, window=ones_window, dim=-1)

        # # Quick and dirty plot of gamma0wm and windowed_data
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(12, 6))
        # plt.plot(gamma0wm.cpu().numpy(), label='gamma0wm', alpha=0.8)
        # plt.plot(wgamma.squeeze().cpu().numpy(), label='windowed_data', alpha=0.8)
        # plt.plot(wgamma2.squeeze().cpu().numpy(), label='windowed_data', alpha=0.8)
        # plt.xlabel('Frame Index')
        # plt.ylabel('Value')
        # plt.title('Gamma0wm and Windowed Data')
        # plt.legend()
        # plt.grid()
        # plt.show()
        # plt.savefig('/data2/Henri/journal2_CB/Playground/gamma0wm_windowed_data.png')

        shifted_data = torch.cat([torch.ones_like(wgamma[:1]), wgamma[:-1]], dim=-1)
        binary_data = (wgamma > threshold).to(torch.float64) * (
            shifted_data < threshold
        ).to(torch.float64)
        binary_data[..., :window_size] = 0

        idx = 0
        while idx < len(binary_data):
            if binary_data[idx] == 1:
                binary_data[idx + 1 : idx + 20] = 0
                idx += 20
            else:
                idx += 1

        spk_starts = torch.cat(
            [
                torch.tensor([0]),
                torch.linspace(R.shape[-3] - 3, R.shape[-3], 4, dtype=torch.int32),
            ],
            dim=-1,
        )
        estspks = torch.argwhere(binary_data == 1).squeeze()
        spk_starts[1 : min(estspks.shape[-1], 3) + 1] = estspks[
            : min(estspks.shape[-1], 3)
        ]

        print(spk_starts)
        return spk_starts, gamma0wm


class Signal:

    def __init__(
        self,
        files: list[list[str]],
        start_times,
        signal_len: float,
        transform: STFTtransform = STFTtransform(),
        sampling_frequency=16e3,
        device: str = "cuda:0",
        dtype=torch.float64,
    ) -> None:
        self.fs = sampling_frequency
        self.num_sources = min(len(files[0]), len(files[1]))
        self.clean_speeches = []
        self.RIRs = []
        clean_signals = []
        RIRs_data = []
        RIRs_direct_data = []
        for source in range(self.num_sources):
            self.clean_speeches += [
                Clean_speech(
                    files[0][source],
                    start_times[source],
                    signal_len=signal_len,
                    device=device,
                    dtype=dtype,
                )
            ]
            self.RIRs += [RIR(files[1][source], device=device, dtype=dtype)]
            clean_signals += [self.clean_speeches[source].data]
            RIRs_data += [self.RIRs[source].data]
            RIRs_direct_data += [self.RIRs[source].direct()[0]]
        self.clean_signals = torch.stack(clean_signals)[:, None, :]
        self.signal_length = self.clean_signals.shape[-1]
        self.RIRs_data = torch.stack(RIRs_data)
        self.RIRs_direct_data = torch.stack(zeropad2fitdims(RIRs_direct_data))
        self.source_signals = convolve_clean2microphone(
            clean=self.clean_signals, rirdata=self.RIRs_data
        )
        self.ref_signals = convolve_clean2microphone(
            clean=self.clean_signals, rirdata=self.RIRs_direct_data
        )  # TODO: do shadow filtering with ref_sigs
        self.noise_signal = Noise(
            files[2], signal_len=signal_len, device=device, dtype=dtype
        )
        self.source_activity = torch.cat(
            [
                self.noise_signal.data.new_ones(1, 1, self.signal_length, dtype=dtype),
                vad_opt(self.source_signals, self.fs, thr=-30, min_on=50e-3),
            ],
            dim=0,
        )
        self.signal_components, norm_factors = normalize_components(
            torch.cat([self.noise_signal.data[None, :, :], self.source_signals], dim=0),
            self.source_activity,
        )
        self.ref_signals = self.ref_signals * norm_factors[1:]
        self.clean_signals = self.clean_signals * norm_factors[1:]
        self.stft_data = transform.encode(self.signal_components)
        self.info = {
            "RIRs": [
                path.basename(self.RIRs[source].filename)
                for source in range(self.num_sources)
            ],
            "cleanSpeeches": [
                path.basename(self.clean_speeches[source].filename)
                for source in range(self.num_sources)
            ],
            "noise": self.noise_signal.noise_type,
        }

    def mixSTFT(self, SNRs=[0, 0, 0, 0]):
        scaled_components = self.stft_data * db2amp(
            torch.tensor(SNRs, device=self.stft_data.device, dtype=self.stft_data.dtype)
        ).view(-1, 1, 1, 1)
        return torch.cat(
            [torch.sum(scaled_components, dim=0, keepdim=True), scaled_components],
            dim=0,
        )

    def plot(self):
        fig, axs = plt.subplots(self.num_sources + 2, 1, figsize=(16, 10))

        # Create time axis based on sampling frequency
        time_axis = np.arange(self.signal_length) / self.fs

        for sig_comp_idx in range(self.num_sources + 2):
            label = (
                "noisy"
                if sig_comp_idx == 0
                else "noise" if sig_comp_idx == 1 else "source" + str(sig_comp_idx - 1)
            )

            # Plot noisy signal (sum of all components)
            if sig_comp_idx == 0:
                axs[sig_comp_idx].plot(
                    time_axis,
                    self.signal_components.sum(dim=-3).squeeze().mT.cpu().numpy(),
                    label=label,
                )

            # Plot noise signal
            elif sig_comp_idx == 1:
                axs[sig_comp_idx].plot(
                    time_axis,
                    self.signal_components[sig_comp_idx - 1].squeeze().mT.cpu().numpy(),
                    label=label,
                )

            # Plot each source signal and overlay the source activity
            else:
                axs[sig_comp_idx].plot(
                    time_axis,
                    self.signal_components[sig_comp_idx - 1].squeeze().mT.cpu().numpy(),
                    label=label,
                )

                # Create a second y-axis for activity
                axy = axs[sig_comp_idx].twinx()
                activity = (
                    self.source_activity[sig_comp_idx - 1].squeeze().cpu().numpy()
                )  # Activity corresponding to the current source
                axy.plot(time_axis, activity, label="activity", color="r")

            axs[sig_comp_idx].set_xlabel("Time [s]")
            axs[sig_comp_idx].set_ylabel(label)
            # axs[sig_comp_idx].legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        plt.savefig("Signal.png")


class RTFestimator:
    # @profile
    def __init__(
        self,
        noisyCovMat: torch.Tensor,
        noiseCovMat: torch.Tensor,
        oldRTFvecs: torch.Tensor,
        calc_method=None,
        methods=["BOPr", "BOPo", "CB", "C"],
        NHmodes=["no", "subtract", "whiten"],
    ) -> None:
        self.Ry = makeHermitian(noisyCovMat)
        self.Rn = makeHermitian(noiseCovMat)
        self.Gi = oldRTFvecs
        self.M = self.Gi.shape[-2]
        self.N = self.Gi.shape[-1] + 1
        self.Na = self.M - self.N
        self.Rs = noise_subtraction(subtractingCovMat=self.Rn, covMat=self.Ry)
        self.Rnsqrt, self.Rw, self.Gw = noise_whitening(
            whiteningCovMat=self.Rn, covMat=self.Ry, RTFvecs=self.Gi[:, -1][:, None]
        )
        self.R = makeMatrixUnitNorm(torch.stack([self.Ry, self.Rs, self.Rw]))
        self.G = torch.cat([self.Gi[:, :2], self.Gw], dim=1).repeat(
            1, 1, 1, 1, 1, self.R.shape[-3], 1, 1
        )
        self.Ga = torch.stack(
            [
                torch.cat(
                    [
                        self.G[0],
                        randdir(
                            self.G.shape[1:-1] + (self.Na,),
                            device=self.G.device,
                            dtype=self.G.dtype,
                        ),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        self.G[1],
                        characteristic_subspace_h(self.R, order=range(-self.Na, 0)),
                    ],
                    dim=-1,
                ),
            ]
        )
        self.hBOP = (
            self.BOPoptimize() if calc_method == "gradient" else self.BOPclosedForm()
        )
        self.RTFest = torch.cat(
            [self.hBOP[:, :2], makeVectorUnitNorm(self.Rnsqrt @ self.hBOP[:, [-1]])],
            dim=1,
        )
        freq = 50
        frame = 40
        # self.plotCostFun(self.R[0,0,0,freq,frame,:,:], self.G[0,0,0,0,freq,frame,:,:], self.hBOP[0,0,0,0,freq,frame,:,:])
        # t=5

    def BOPcostfun(
        self, R: torch.Tensor, G: torch.Tensor, h: torch.Tensor, comp_grad: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        P_Gh = oblique_projection(G, h)
        cost = torch.real(trace(P_Gh @ R @ P_Gh.mH))
        if not comp_grad:
            return cost
        else:
            return cost, self.BOPgradient(R, G, h)

    def BOPgradient(
        self, R: torch.Tensor, G: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        P_Gh = oblique_projection(G, h)
        summand_1 = -(
            P_Gh.mH
            @ (
                P_Gh
                @ (
                    R
                    @ (
                        (
                            torch.eye(
                                *P_Gh.shape[-2:], device=P_Gh.device, dtype=P_Gh.dtype
                            )
                            - P_Gh.mH
                        )
                        @ (h / (torch.linalg.vector_norm(h, dim=-2, keepdim=True) ** 2))
                    )
                )
            )
        )
        if G.shape[-1] == G.shape[-2] - 1:
            return summand_1
        else:
            Porth_GH = orthogonal_projection(generalized_cat([G, h], dim=-1))
            Ph = orthogonal_projection(h)
            summand_2 = -(
                Porth_GH
                @ (
                    R
                    @ (
                        P_Gh.mH
                        @ (
                            G
                            @ torch.linalg.solve(
                                G.mH @ (Ph @ G),
                                (
                                    G.mH
                                    @ (
                                        h
                                        / torch.linalg.vector_norm(
                                            h, dim=-2, keepdim=True
                                        )
                                        ** 2
                                    )
                                ),
                            )
                        )
                    )
                )
            )
            return summand_1 + summand_2

    def BOPoptimize(self, convergence_threshold=1e-6, maxIter=1000):
        h = randdir(
            self.Ga.shape[:-1] + (1,), device=self.Ga.device, dtype=self.Ga.dtype
        )
        optimizer = torch.optim.Rprop([h])
        for iteration in range(maxIter):
            hold = h.clone()
            h.grad = self.BOPgradient(self.R, self.Ga, h)
            optimizer.step()  # Update the vector variable
            if torch.all(
                torch.linalg.vector_norm(hold - h, dim=-2, keepdim=True)
                / torch.linalg.vector_norm(hold, dim=-2, keepdim=True)
                < convergence_threshold
            ):
                print(iteration)
                break
        return h.requires_grad_(False)

    def BOPoptimize2(self, convergence_threshold=1e-6, maxIter=1000):
        grad_perf = []
        for av in [0, 1]:
            grad_perf2 = []
            for nh in [0, 1, 2]:
                converged = False
                n_init = 1
                lr = 1e0
                maxIter = 100
                while not converged:
                    h = randdir(
                        self.Ga.shape[:-1] + (1,),
                        device=self.Ga.device,
                        dtype=self.Ga.dtype,
                    )[av : av + 1, nh : nh + 1]
                    optimizer = torch.optim.SGD([h], lr=lr)
                    hcf = self.BOPclosedForm()[av : av + 1, nh : nh + 1]
                    HA = [hermitian_angle(h, hcf)]
                    C = [
                        self.BOPcostfun(
                            self.R[nh : nh + 1], self.Ga[av : av + 1, nh : nh + 1], h
                        )
                    ]
                    C_cf = self.BOPcostfun(
                        self.R[nh : nh + 1], self.Ga[av : av + 1, nh : nh + 1], hcf
                    )
                    CC_n = []
                    CC_c = []
                    for iteration in range(maxIter):
                        hold = h.clone()
                        h.grad = self.BOPgradient(
                            self.R[nh : nh + 1], self.Ga[av : av + 1, nh : nh + 1], h
                        )
                        optimizer.step()  # Update the vector variable
                        # makeVectorUnitNorm_inPlace(h)
                        HA += [hermitian_angle(h, hcf)]
                        C += [
                            self.BOPcostfun(
                                self.R[nh : nh + 1],
                                self.Ga[av : av + 1, nh : nh + 1],
                                h,
                            )
                        ]
                        CC_n += [
                            torch.linalg.vector_norm(hold - h, dim=-2, keepdim=True)
                            / torch.linalg.vector_norm(hold, dim=-2, keepdim=True)
                        ]
                        CC_c += [
                            (
                                self.BOPcostfun(
                                    self.R[nh : nh + 1],
                                    self.Ga[av : av + 1, nh : nh + 1],
                                    h,
                                )
                                - self.BOPcostfun(
                                    self.R[nh : nh + 1],
                                    self.Ga[av : av + 1, nh : nh + 1],
                                    hold,
                                )
                            )
                            / self.BOPcostfun(
                                self.R[nh : nh + 1],
                                self.Ga[av : av + 1, nh : nh + 1],
                                hold,
                            )
                        ]

                        # print('Iter:' + str(iteration) + ' hdif: '  + str(values.item())
                        #                                + ' HA: ' + str(HA[-1].item()/torch.pi*180)
                        #                                #+ ' conv: ' + str(torch.sum(values < convergence_threshold).item()) + '/' + str(values.numel())
                        #                                + ' cost: ' + str(self.BOPcostfun(self.R, self.Ga, h).item()) + '/' + str(self.BOPcostfun(self.R, self.Ga, hcf).item())
                        #                                + ' cdif: ' + str(costdif.item())
                        #                                + ' norm: ' + str(torch.linalg.vector_norm(h).item()))
                        if torch.all(
                            torch.linalg.vector_norm(hold - h, dim=-2, keepdim=True)
                            / torch.linalg.vector_norm(hold, dim=-2, keepdim=True)
                            < convergence_threshold
                        ):

                            if iteration < 3:
                                converged = False
                            else:
                                converged = True
                                print("I:", iteration, " - n_init: ", n_init)
                            break
                        elif iteration == maxIter - 1:
                            n_init += 1
                            lr = max(1e0, lr / 2)
                            maxIter += 100
                            # print('I:', iteration, ' - n_init: ', n_init)
                grad_perf2 += [[HA, C, CC_n, CC_c, C_cf, n_init]]
            grad_perf += [grad_perf2]
        return grad_perf

    # @profile
    def BOPclosedForm(self) -> torch.Tensor:
        Glist = [self.Ga, self.G[2:]]
        hlist = []
        for G in Glist:
            if (
                G.shape[-1] == G.shape[-2] - 1
            ):  # "G should be of dimension M x M-1, where M is the number of microphones!"
                hlist += [
                    makeVectorUnitNorm(
                        torch.mean(
                            self.R @ orthogonal_projection(G), dim=-1, keepdim=True
                        )
                    )
                ]
            else:
                hlist += [
                    characteristic_subspace(
                        self.R @ orthogonal_projection(G), left=True
                    )
                ]
        return torch.cat(hlist, dim=0)

    def plotCostFun(self, R, G, h):

        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Computer Modern"

        h = characteristic_subspace(R @ orthogonal_projection(G), left=True)
        Gh = torch.cat([G, h], dim=-1)
        Ghorth = orthogonal_complement(Gh)[..., [0]]
        Gmain = peigvech(parallel_projection(G))
        N = 1000
        xlim = [0.01, 0.99]
        ylim = [0.01, 0.99]
        iG2Gho = slerp(
            Gmain, Ghorth, torch.linspace(*xlim, N, device=h.device, dtype=h.dtype)
        )
        iG2Gho = iG2Gho.permute(-1, *range(iG2Gho.ndimension() - 1))[..., None]
        RTFmesh = slerp(
            h, iG2Gho, torch.linspace(*ylim, N, device=h.device, dtype=h.dtype)
        )
        RTFmesh = RTFmesh.permute(-1, *range(RTFmesh.ndimension() - 1))[..., None]
        Gr = torch.cat(
            [G, randdir(G.shape[0:-1] + (self.Na,), device=G.device, dtype=G.dtype)],
            dim=-1,
        )
        Go = torch.cat(
            [G, characteristic_subspace_h(R, order=range(-self.Na, 0))], dim=-1
        )
        Cost, Grad = self.BOPcostfun(R, G, RTFmesh, comp_grad=True)
        Costr, Gradr = self.BOPcostfun(R, Gr, RTFmesh, comp_grad=True)
        Costo, Grado = self.BOPcostfun(R, Go, RTFmesh, comp_grad=True)

        # Create grid for plotting
        x = np.linspace(*xlim, N)  # Adjust as needed for your domain
        y = np.linspace(*ylim, N)
        X, Y = np.meshgrid(x, y)

        # Prepare figure
        fig = plt.figure(figsize=(16, 9), dpi=600)

        # Function to add surface and contour plots
        markersize = 1000

        def add_plots(fig, position, Z, plot_type, title, handles=None):
            if plot_type == "surface":
                ax = fig.add_subplot(2, 3, position, projection="3d")
                surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
                h_line = Z[
                    np.argmin(np.abs(y - 0)), :
                ].flatten()  # Values of Z along y=0
                hline = ax.plot(
                    x,
                    np.zeros_like(x),
                    h_line,
                    color="orange",
                    linewidth=5,
                    label="new RTF $\\mathbf{h}$",
                )
                G_cross = Z[np.argmin(np.abs(y - 1)), np.argmin(np.abs(x - 0))]
                Gdot = ax.scatter(
                    0,
                    1,
                    G_cross,
                    color="red",
                    marker=".",
                    s=markersize,
                    label="old RTF $\\mathbf{G}$",
                )
                Ghorth_cross = Z[np.argmin(np.abs(y - 1)), np.argmin(np.abs(x - 1))]
                Ghorthdot = ax.scatter(
                    1,
                    1,
                    Ghorth_cross,
                    color="purple",
                    marker=".",
                    s=markersize,
                    label="orthogonal subspace",
                )
                ax.set_title(title, fontsize=15)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                fig.colorbar(surf, ax=ax, shrink=0.5)
            elif plot_type == "contour":
                ax = fig.add_subplot(2, 3, position)
                contour = ax.contourf(X, Y, Z, levels=21, cmap="viridis")
                hline = ax.plot(
                    x,
                    np.zeros_like(x),
                    color="orange",
                    linewidth=10,
                    label="new RTF $\\mathbf{h}$",
                )
                Gdot = ax.scatter(
                    0,
                    1,
                    color="red",
                    marker=".",
                    s=markersize,
                    label="old RTF $\\mathbf{G}$",
                )
                Ghorthdot = ax.scatter(
                    1,
                    1,
                    color="purple",
                    marker=".",
                    s=markersize,
                    label="orthogonal subspace",
                )
                ax.set_title(title, fontsize=15)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                fig.colorbar(contour, ax=ax)
            if handles is not None:
                handles.append(hline[0])
                handles.append(Gdot)
                handles.append(Ghorthdot)

        # Add surface plots
        add_plots(fig, 1, pow2db(Cost).squeeze().cpu().numpy(), "surface", "Cost")
        add_plots(
            fig,
            2,
            pow2db(Costr).squeeze().cpu().numpy(),
            "surface",
            "Cost with random additional vectors $\\mathbf{A}$",
        )
        add_plots(
            fig,
            3,
            pow2db(Costo).squeeze().cpu().numpy(),
            "surface",
            "Cost with orthogonal additional vectors $\\mathbf{A}$",
        )

        # Add contour plots
        add_plots(fig, 4, pow2db(Cost).squeeze().cpu().numpy(), "contour", "Cost")
        add_plots(
            fig,
            5,
            pow2db(Costr).squeeze().cpu().numpy(),
            "contour",
            "Cost with random additional vectors $\\mathbf{A}$",
        )
        handles = []
        add_plots(
            fig,
            6,
            pow2db(Costo).squeeze().cpu().numpy(),
            "contour",
            "Cost with orthogonal additional vectors $\\mathbf{A}$",
            handles,
        )

        fig.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(1, 0.6),
            title="\\textbf{Legend}",
            fontsize=15,
        )

        # Adjust layout
        plt.tight_layout(rect=(0, 0, 0.9, 1))  # Leave space on the right for the legend

        # Save or display with enough space
        plt.subplots_adjust(
            right=0.85
        )  # Adjust figure to include legend in the white space
        plt.show()
        plt.savefig("Playground/Test.png")

        fig2 = plt.figure(figsize=(16, 9), dpi=600)

        # Add surface plots
        add_plots(
            fig2,
            1,
            pow2db(torch.linalg.vector_norm(Grad, dim=-2)).squeeze().cpu().numpy(),
            "surface",
            "Cost Surface",
        )
        add_plots(
            fig2,
            2,
            pow2db(torch.linalg.vector_norm(Gradr, dim=-2)).squeeze().cpu().numpy(),
            "surface",
            "Costr Surface",
        )
        add_plots(
            fig2,
            3,
            pow2db(torch.linalg.vector_norm(Grado, dim=-2)).squeeze().cpu().numpy(),
            "surface",
            "Costo Surface",
        )

        # Add contour plots
        add_plots(
            fig2,
            4,
            pow2db(torch.linalg.vector_norm(Grad, dim=-2)).squeeze().cpu().numpy(),
            "contour",
            "Cost Contour",
        )
        add_plots(
            fig2,
            5,
            pow2db(torch.linalg.vector_norm(Gradr, dim=-2)).squeeze().cpu().numpy(),
            "contour",
            "Costr Contour",
        )
        handles = []
        add_plots(
            fig2,
            6,
            pow2db(torch.linalg.vector_norm(Grado, dim=-2)).squeeze().cpu().numpy(),
            "contour",
            "Costo Contour",
            handles,
        )

        fig2.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(1, 0.6),
            title="Legend",
            fontsize=15,
        )

        # Adjust layout
        plt.tight_layout(rect=(0, 0, 0.9, 1))  # Leave space on the right for the legend

        # Save or display with enough space
        plt.subplots_adjust(
            right=0.85
        )  # Adjust figure to include legend in the white space
        plt.show()
        plt.savefig("Playground/Test2.png")





def ps(tensorlist: List[torch.Tensor]):
    for t in tensorlist:
        print(t.shape)
    print("---")


def checkRTFestimationMethods(method, M=None, N=None):
    if M is None:
        M = 8
    if N is None:
        N = 3
    h = randdir(M, 1)
    G = randdir(1, 1, 1, 1, M, N - 1)
    Rn = randdir(1, 1, 1, 1, M, M)
    Rn = makeHermitian(Rn @ Rn.mH)
    Ry = makeHermitian(h @ h.mH + G @ G.mH + Rn)
    G = G.repeat(3, 3, 1, 1, 1, 1, 1, 1)

    h_est = method(Ry, Rn, G).squeeze()[..., None]
    print("h:", h)
    print("h_est:", h_est)
    print("factor:", h_est / h)
    print("HA:", hermitian_angle(h, h_est))

    return None


# %% Utility Functions


def load_audio(
    filename: str,
    sampling_frequency: int = 16000,
    device: Union[str, torch.device] = "cuda:0",
    dtype=torch.float64,
) -> tuple[torch.Tensor, float]:
    """
    Loads an audio file, converts it to a tensor, normalizes by subtracting the mean,
    and transfers it to the specified device and data type.

    Args:
        filename (str): The path to the audio file to be loaded.
        sampling_frequency (int, optional): The target sampling frequency for the audio file. Defaults to 16000 Hz.
        device (str, optional): The device where the audio tensor will be stored (e.g., 'cuda:0' for a GPU). Defaults to 'cuda:0'.
        dtype (torch.dtype, optional): The data type to which the tensor will be converted. Defaults to torch.float64.

    Returns:
        tuple[torch.Tensor, float]: A tuple containing:
            - A tensor of the audio data (mean normalized), transferred to the specified device and dtype.
            - The actual sampling frequency of the loaded audio file.
    """
    # Step 1: Load the audio file using torchaudio.
    waveform, actual_sampling_rate = torchaudio.load(uri=filename)

    # Step 2: Resample the audio if the sampling rate is not the target frequency.
    if actual_sampling_rate != sampling_frequency:
        resampler = torchaudio.transforms.Resample(
            orig_freq=actual_sampling_rate,
            new_freq=sampling_frequency,
            dtype=waveform.dtype,
        )
        waveform = resampler(waveform)
        actual_sampling_rate = sampling_frequency

    # Step 3: Convert the it to the specified device and dtype.
    audio_tensor = waveform.to(device=device, dtype=dtype)

    # Step 4: Normalize the audio by subtracting its mean along the last axis (usually the time axis).
    # This ensures the audio has zero mean, removing any DC offset.
    mean_normalized_audio = audio_tensor - audio_tensor.mean(dim=-1, keepdim=True)

    # Step 5: Return the normalized audio tensor and the actual sampling frequency.
    return mean_normalized_audio, actual_sampling_rate


def slice2segments(
    signal: torch.Tensor, segment_borders: torch.Tensor
) -> list[torch.Tensor]:
    """
    Splits a signal tensor into segments based on provided segment borders.

    Args:
        signal (torch.Tensor): The input signal tensor to be split. This can have any number of dimensions, but
                               the last dimension is assumed to represent time or the sequence to be split.
        segment_borders (torch.Tensor): A 1D tensor containing the segment start and end indices. The borders
                                        define the points in time (or along the sequence) where the segments begin and end.
                                        The length of this tensor should be at least 2 (start and end).

    Returns:
        list[torch.Tensor]: A list of tensor segments, where each segment is sliced from `signal` according to the
                            indices in `segment_borders`. Each segment corresponds to a slice from `seg_start` to `seg_end`
                            along the last dimension of `signal`.
    """

    # Use list comprehension to iterate over pairs of consecutive segment borders (start, end).
    # Each segment is sliced from the `signal` tensor using the start and end indices, slicing along the last dimension.
    # The signal[..., seg_start:seg_end] syntax ensures that slicing occurs on the last dimension, regardless of
    # how many other dimensions the tensor has.
    return [
        signal[..., seg_start:seg_end]
        for seg_start, seg_end in zip(segment_borders[:-1], segment_borders[1:])
    ]


def print_shapes(nested_list, indent=0):
    """
    Recursively prints the shapes of tensors in a nested list structure.

    Args:
        nested_list: A nested list structure containing tensors or other lists.
        indent: The current indentation level for tree-like formatting.
    """
    if isinstance(nested_list, torch.Tensor):
        print(" " * indent + f"Tensor: {nested_list.shape}")
        return
    else:
        for i, item in enumerate(nested_list):
            if isinstance(item, list):
                print(" " * indent + f"List[{i}]:")
                print_shapes(item, indent + 4)  # Increase indentation for nested lists
            elif isinstance(item, tuple):
                print(" " * indent + f"Tuple[{i}]:")
                print_shapes(item, indent + 4)  # Increase indentation for nested tuples
            elif isinstance(item, torch.Tensor):
                print(" " * indent + f"Tensor[{i}]: {item.shape}")
            else:
                print(" " * indent + f"Item[{i}]: {type(item)} (not a tensor or list)")


def print_dtypes(nested_list, indent=0):
    """
    Recursively prints the shapes of tensors in a nested list structure.

    Args:
        nested_list: A nested list structure containing tensors or other lists.
        indent: The current indentation level for tree-like formatting.
    """
    if isinstance(nested_list, torch.Tensor):
        print(" " * indent + f"Tensor: {nested_list.dtype}")
        return
    else:
        for i, item in enumerate(nested_list):
            if isinstance(item, list):
                print(" " * indent + f"List[{i}]:")
                print_dtypes(item, indent + 4)  # Increase indentation for nested lists
            elif isinstance(item, tuple):
                print(" " * indent + f"Tuple[{i}]:")
                print_dtypes(item, indent + 4)  # Increase indentation for nested tuples
            elif isinstance(item, torch.Tensor):
                print(" " * indent + f"Tensor[{i}]: {item.dtype}")
            else:
                print(" " * indent + f"Item[{i}]: {type(item)} (not a tensor or list)")


def flatten_and_stack_tensorlist(
    tensorlist: list[torch.Tensor], num_dims: int = 2, verbose: bool = False
) -> tuple[torch.Tensor, list[tuple]]:
    """
    Flattens and stacks a list of tensors along all but the last two dims.
    Returns stacked tensor and list of original shapes.
    """
    shapes = [t.shape for t in tensorlist]
    reshaped = []
    for t in tensorlist:
        # Flatten all dims except last two
        new_shape = (-1,) + t.shape[-num_dims:]
        reshaped.append(t.reshape(new_shape))
        if verbose:
            print(f"Original shape: {t.shape} -> Flattened shape: {reshaped[-1].shape}")
    stacked = torch.cat(reshaped, dim=0)
    if verbose:
        print(f"Stacked shape: {stacked.shape}")
    return stacked, shapes


def unstack_and_reshape_tensorlist(
    stacked: torch.Tensor, shapes: list[tuple], num_dims: int = 2, verbose: bool = False
) -> list[torch.Tensor]:
    """
    Splits and reshapes the stacked tensor back into the original list of tensors.
    """
    out = []
    idx = 0
    for shape in shapes:
        n = int(torch.prod(torch.tensor(shape[:-num_dims])))
        t = stacked[idx : idx + n].reshape(shape[:-num_dims] + stacked.shape[1:])
        out.append(t)
        if verbose:
            print(f"Recovered shape: {t.shape}")
        idx += n
    return out
