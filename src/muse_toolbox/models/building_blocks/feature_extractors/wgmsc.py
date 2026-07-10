import torch
from muse_toolbox.utils import (
    STFTtransform,
    smoothCovarianceMatrix,
    windowedCovarianceMatrix,
    noise_whitening_robust,
    coherenceMatrix,
    gmsc,
    trace,
    wmean,
    regularize,
    get_real_dtype,
)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .base_feature import BaseFeatureExtractor
import os


class WGMSC_Feature_Extractor(BaseFeatureExtractor):
    """
    Implements the Whitened Generalized Magnitude-Squared Coherence (WGMSC)
    as a building block for the Coherence-based Source Activity Detector (COSAD).

    This module processes a multi-channel audio mixture to compute the
    spatial Whitened Generalized Magnitude-Squared Coherence (WGMSC)
    of the sound field. The WGMSC is a measure of spatial coherence that is
    robust to noise, making it suitable for detecting the presence of coherent
    sound sources.
    """

    def __init__(
        self,
        transform: STFTtransform,
        smoothing_time_constant: float,  # [s]
        whitening_time_constant: float,  # [s]
        smoothing_time_constant_rev: float,  # [s]
        whitening_time_constant_rev: float,  # [s]
        rev_features: bool = True,
        wideband_features: bool = False,
    ) -> None:
        """
        Initializes the COSAD_WGMSC module.

        Args:
            transform (STFTtransform): An STFT transformation object to convert
                time-domain signals to the time-frequency domain. It provides
                parameters like frame shift and sampling frequency.
            smoothing_time_constant (float, optional): Time constant in seconds for
                recursively smoothing the covariance matrices. A larger value
                results in smoother estimates but with more latency. Defaults to 1.0.
            whitening_time_constant (float, optional): Time constant in seconds for
                estimating the noise covariance matrix for whitening. This defines
                the look-back window for noise estimation. Defaults to 1.0.
        """
        super().__init__(transform=transform)
        self.transform = transform
        self.smoothing_time_constant = smoothing_time_constant
        self.whitening_time_constant = whitening_time_constant
        self.smoothing_time_constant_rev = smoothing_time_constant_rev
        self.whitening_time_constant_rev = whitening_time_constant_rev
        self.rev_features = rev_features
        self.wideband_features = wideband_features

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Prints the parameters of the module in a structured, indented format.

        Args:
            indent (str, optional): A string to prepend to each line for indentation.
                                    Defaults to "".
        """
        print(f"{indent}{self.__class__.__name__} Parameters:")
        print(f"{indent}  Smoothing Time Constant: {self.smoothing_time_constant} s")
        print(f"{indent}  Whitening Time Constant: {self.whitening_time_constant} s")
        print(
            f"{indent}  Smoothing Time Constant (Rev): {self.smoothing_time_constant_rev} s"
        )
        print(
            f"{indent}  Whitening Time Constant (Rev): {self.whitening_time_constant_rev} s"
        )
        print(f"{indent}  Reverse Features Enabled: {self.rev_features}")
        print(
            f"{indent}  Features Type: {'wideband' if self.wideband_features else 'narrowband'}"
        )

    @property
    def signature(self) -> str:
        freqstr = "wb" if self.wideband_features else "nb"
        signature = (
            f"WGMSC_{freqstr}"
            f"_fl{self.transform.frame_length}_fs{self.transform.frame_shift}"
            f"_sf{self.transform.sampling_frequency}_win{self.transform.window_type}"
            f"_stc{self.smoothing_time_constant}_wtc{self.whitening_time_constant}"
        )
        if self.rev_features:
            signature += f"_stcr{self.smoothing_time_constant_rev}_wtcr{self.whitening_time_constant_rev}"
        return signature

    @property
    def feature_dim(self) -> int:
        num_freq_bins = self.transform.num_freq_bins
        dim = 1 if self.wideband_features else num_freq_bins
        if self.rev_features:
            dim *= 2
        return dim

    @property
    def is_trainable(self) -> bool:
        """
        Indicates whether this feature extractor contains learnable parameters.
        """
        return False

    def get_config(self) -> dict:
        return {
            "smoothing_time_constant": self.smoothing_time_constant,
            "whitening_time_constant": self.whitening_time_constant,
            "smoothing_time_constant_rev": self.smoothing_time_constant_rev,
            "whitening_time_constant_rev": self.whitening_time_constant_rev,
            "rev_features": self.rev_features,
            "wideband_features": self.wideband_features,
        }

    def forward_stft(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to compute WGMSC from a multi-channel mixture.

        Args:
            batch (torch.Tensor): The input mixture signal. Shape: (B, M, N)

        Returns:
            torch.Tensor: The output features. Shape: (B, J, T)
        """

        # Step 1: Transform the time-domain signal to the STFT domain.
        # STFTtransform usually handles (B, M, N) -> (B, F, T, M) or similar
        stft_mix = batch

        # Step 2: Calculate the narrowband coherence and whitened covariance.
        # The internal methods are already vectorized for batch processing.
        wgmsc_narrowband, Rw, wgmsc_narrowband_rev, Rw_rev = (
            self._white_gmsc_narrowband(stft_mix)
        )

        # Prepare Output
        if self.wideband_features:
            # Output shape: (B, 1, T)
            wgmsc_wideband = self._combine_frequencies(wgmsc_narrowband, Rw)
            fwd = wgmsc_wideband[..., 0, 0]  # (B, 1, T)
            if self.rev_features:
                wgmsc_wideband_rev = self._combine_frequencies(
                    wgmsc_narrowband_rev, Rw_rev
                )
                rev = wgmsc_wideband_rev[..., 0, 0]  # (B, 1, T)
                output = torch.cat([fwd, rev], dim=1)  # (B, 2, T)
            else:
                output = fwd
        else:
            # Output shape: (B, F, T)
            fwd = wgmsc_narrowband[..., 0, 0]  # (B, F, T)
            if self.rev_features:
                rev = wgmsc_narrowband_rev[..., 0, 0]  # (B, F, T)
                output = torch.cat([fwd, rev], dim=-2)  # (B, 2F, T)
            else:
                output = fwd

        # if False:  # DEBUGGING | PROTOTYPING
        #     print("DEBUGGING | PROTOTYPING")
        #     self._plot_wgmsc(
        #         wgmsc_narrowband,
        #         wgmsc_wideband,
        #         batch_idx=0,
        #         output_path="Playground/Awgmsc.png",
        #     )

        #     self._plot_wgmsc(
        #         wgmsc_narrowband_rev,
        #         exp_windowing(
        #             wgmsc_wideband_rev,
        #             smoothing_factor=self.transform.timeConstant2smoothingFactor(0.5),
        #             dim=-3,
        #         ),
        #         batch_idx=0,
        #         output_path="Playground/Awgmsc_rev.png",
        #     )

        # self._plot_wgmsc(
        #     wgmsc_narrowband,
        #     wgmsc_wideband,
        #     batch_idx=1,
        #     output_path="Playground/Bwgmsc.png",
        # )

        # self._plot_wgmsc(
        #     wgmsc_narrowband_rev,
        #     wgmsc_wideband_rev,
        #     batch_idx=1,
        #     output_path="Playground/Bwgmsc_rev.png",
        # )

        return output

    def _white_gmsc_narrowband(
        self, mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the Whitened Generalized Magnitude-Squared Coherence (WGMSC)
        for each frequency bin.

        This involves smoothing the mixture's covariance matrix, estimating a
        noise covariance from past frames, whitening the smoothed covariance,
        and finally computing the GMSC from the whitened result.

        Args:
            mix (torch.Tensor): The input mixture signal in the STFT domain.
                                Shape: (B, ..., F, M, T).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - gamma (torch.Tensor): The narrowband GMSC values. Shape: (B, ..., F, 1, T).
                - Rw (torch.Tensor): The whitened covariance matrix. Shape: (B, ..., F, T, M, M).
        """
        M = mix.shape[-2]
        # Calculate the recursively smoothed covariance matrix of the mixture.
        Ry = regularize(
            smoothCovarianceMatrix(
                mix,
                smoothing_factor=self.transform.timeConstant2smoothingFactor(
                    self.smoothing_time_constant
                ),
            ),
            reg_factor=1e-6,
        )

        # Regularize the covariance matrix to ensure it is well-conditioned for inversion.
        # Ry[..., :M, :, :] = regularize(Ry[..., :M, :, :], 1e-1)
        # Determine the indices for causal estimation of the noise covariance matrix.
        # For each frame, this creates a look-back window into the past, the size
        # of which is determined by `whitening_time_constant`.
        whiteidx = torch.arange(0, Ry.shape[-3], device=Ry.device, dtype=torch.long)
        lookback_frames = self.transform.times2frames(self.whitening_time_constant)
        whiteidx = whiteidx - torch.min(whiteidx // 2, lookback_frames)

        # Estimate the noise covariance matrix from past frames.
        Rv = Ry[..., whiteidx, :, :]

        # Rv[..., :M, :, :] = regularize(Rv[..., :M, :, :], 1e-1)
        # starting index
        start = 0  # 2 * (M - 1)
        # Perform noise whitening to get the whitened covariance (Rw).
        _, Rw, _ = noise_whitening_robust(
            Rv[..., start:, :, :], Ry[..., start:, :, :], subtract_identity=False
        )
        # _, Rw, _ = noise_whitening_noncholesky(Rv, Ry, subtract_identity=False)
        # _, Rw_rev, _ = noise_whitening_noncholesky(Ry, Rv, subtract_identity=False)

        # Calculate the coherence matrix from the whitened signal covariance.
        Cw = coherenceMatrix(Rw)
        # Cw_rev = coherenceMatrix(Rw_rev)
        # Compute the Generalized Magnitude-Squared Coherence (GMSC).
        gamma = gmsc(Cw).to(dtype=get_real_dtype(mix), device=mix.device)
        # gamma_rev = gmsc(Cw_rev)
        # Pad the beginning of gamma and Rw to match the original time dimension length.
        # This is necessary because the first `start` frames were excluded from
        # the whitening process to ensure numerical stability.

        if start != 0:
            # 1. Pad gamma with zeros.
            pad_shape_gamma = list(gamma.shape)
            pad_shape_gamma[-3] = start
            gamma_pad = torch.zeros(
                pad_shape_gamma, device=gamma.device, dtype=gamma.dtype
            )
            gamma = torch.cat([gamma_pad, gamma], dim=-3)

            # 2. Pad Rw with identity matrices.
            pad_shape_rw = list(Rw.shape)
            pad_shape_rw[-3] = start
            # Create a batch of identity matrices for padding
            identity_matrix = torch.eye(M, device=Rw.device, dtype=Rw.dtype)
            # Expand the identity matrix to match the padding shape (B, ..., F, start, M, M)
            rw_pad = identity_matrix.expand(pad_shape_rw)
            Rw = torch.cat([rw_pad, Rw], dim=-3)

        Rw = Rw.to(dtype=mix.dtype, device=mix.device)

        if self.rev_features:
            Ry_rev = regularize(
                windowedCovarianceMatrix(
                    mix,
                    window=torch.ones(
                        int(
                            self.transform.times2frames(
                                self.smoothing_time_constant_rev
                            )
                        ),
                        device=mix.device,
                    ),
                ),
                reg_factor=1e-6,
            )
            lookback_frames_rev = self.transform.times2frames(
                self.whitening_time_constant_rev
            )
            whiteidx_rev = whiteidx - torch.min(whiteidx // 2, lookback_frames_rev)

            # Estimate the noise covariance matrix from past frames.
            Rv_rev = Ry_rev[..., whiteidx_rev, :, :]

            start_rev = 2 * (M - 1)
            # Perform noise whitening to get the whitened covariance (Rw).
            _, Rw_rev, _ = noise_whitening_robust(
                Ry_rev[..., start_rev:, :, :],
                Rv_rev[..., start_rev:, :, :],
                subtract_identity=False,
            )
            # Calculate the coherence matrix from the whitened signal covariance.
            Cw_rev = coherenceMatrix(Rw_rev)
            # Compute the Generalized Magnitude-Squared Coherence (GMSC).
            gamma_rev = gmsc(Cw_rev).to(dtype=get_real_dtype(mix), device=mix.device)

            if start_rev != 0:
                # 1. Pad gamma with zeros.
                pad_shape_gamma_rev = list(gamma_rev.shape)
                pad_shape_gamma_rev[-3] = start_rev
                gamma_pad_rev = torch.zeros(
                    pad_shape_gamma_rev, device=gamma_rev.device, dtype=gamma_rev.dtype
                )
                gamma_rev = torch.cat([gamma_pad_rev, gamma_rev], dim=-3)

                # 2. Pad Rw with identity matrices.
                pad_shape_rw_rev = list(Rw_rev.shape)
                pad_shape_rw_rev[-3] = start_rev
                # Create a batch of identity matrices for padding
                identity_matrix_rev = torch.eye(
                    M, device=Rw_rev.device, dtype=Rw_rev.dtype
                )
                # Expand the identity matrix to match the padding shape (B, ..., F, start, M, M)
                rw_pad_rev = identity_matrix_rev.expand(pad_shape_rw_rev)
                Rw_rev = torch.cat([rw_pad_rev, Rw_rev], dim=-3)

            Rw_rev = Rw_rev.to(dtype=mix.dtype, device=mix.device)

        else:
            gamma_rev = torch.zeros_like(gamma)
            Rw_rev = torch.zeros_like(Rw)

        return gamma, Rw, gamma_rev, Rw_rev

    def _combine_frequencies(
        self, wgmsc_narrowband: torch.Tensor, Rw: torch.Tensor
    ) -> torch.Tensor:
        """
        Combines narrowband coherence values into a single wideband value.

        This is done via a weighted average, where the weights are derived from
        the power (trace) of the whitened signal's covariance matrix at each
        frequency bin. This gives more importance to frequency bins with
        higher energy after whitening.

        Args:
            wgmsc_narrowband (torch.Tensor): The per-frequency coherence values.
            Rw (torch.Tensor): The whitened covariance matrix, used for weighting.

        Returns:
            torch.Tensor: The single wideband coherence value for each time frame.
        """
        # Use the trace of the whitened covariance matrix (i.e., power) as weights.
        weights = trace(Rw).real
        # Compute the weighted mean of the narrowband coherence across the frequency dimension.
        return wmean(
            wgmsc_narrowband,
            dims=-4,  # Frequency dimension
            weights=weights,
        )

    def _plot_wgmsc(
        self,
        wgmsc_narrowband: torch.Tensor,
        wgmsc_wideband: torch.Tensor,
        batch_idx: int,
        output_path: str,
    ):
        """
        Plots the narrowband and wideband WGMSC for a specific batch item.

        The top subplot shows the narrowband WGMSC as a spectrogram, and the bottom
        subplot shows the wideband WGMSC as a time-series plot.

        Args:
            wgmsc_wideband (torch.Tensor): The wideband coherence tensor.
                Shape: (B, 1, T, 1, 1).
            wgmsc_narrowband (torch.Tensor): The narrowband coherence tensor.
                Shape: (B, F, T, 1, 1).
            batch_idx (int): The index of the batch item to plot.
            output_path (str): The path to save the output PNG image.
        """
        # --- 1. Prepare data for plotting ---
        # Select the specified batch item and remove singleton dimensions
        narrowband_data = wgmsc_narrowband[batch_idx].squeeze().cpu().numpy()
        wideband_data = wgmsc_wideband[batch_idx].squeeze().cpu().numpy()

        # --- 2. Get time and frequency axes from the transform object ---
        num_frames = narrowband_data.shape[-1]
        time_axis = self.transform.frames2times(torch.arange(num_frames)).cpu().numpy()
        freq_axis = self.transform.frequencies().cpu().numpy()

        # --- 3. Create the two-panel plot ---
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle(f"WGMSC Analysis (Batch Item {batch_idx})")

        # --- 4. Top Subplot: Narrowband WGMSC Spectrogram ---
        im = ax1.imshow(
            narrowband_data,
            aspect="auto",
            origin="lower",
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        ax1.set_ylabel("Frequency (Hz)")
        # Add padding to the title to lift it above the colorbar.
        # ax1.set_title("Narrowband WGMSC")

        # Create a dedicated axis for a horizontal colorbar above the main plot
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("top", size="7%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Coherence")
        # Position the colorbar ticks and label on top
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

        # --- 5. Bottom Subplot: Wideband WGMSC Time Series ---
        ax2.plot(time_axis, wideband_data)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Coherence")
        # ax2.set_title("Wideband WGMSC")
        ax2.set_ylim(0, 1)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # --- 6. Finalize and save ---
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust for suptitle

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path)
        plt.close(fig)
        print(f"WGMSC plot saved to {output_path}")
