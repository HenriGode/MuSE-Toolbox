"""Utility functions and classes for MuSE-Toolbox."""

from .gen_utils import *
from .math import (
    db2amp, db2pow, amp2db, pow2db, rad2deg, deg2rad,
    hermitian_angle, complex_angle, subspace_angles, atan2, atan3, quadrant,
    covariance_SCM, growing_average_SCM, weighted_SCM, crossCovariance_SCM,
    covariance_Tyler, make2covariance_matrix, make2covariance_matrix_rel_lower_bound,
    spherical2cartesian, cartesian2spherical, slerp, successive_projections, moduloshift,
    cpu_gen_solve, effective_rank, is_hermitian, is_symmetric, is_positive_definite_h,
    is_positive_semi_definite_h, trace, makeHermitian, makeSymmetric,
    make_positive_definite_h, evd2matrix_h, makeMatrixUnitNorm, makeMatricesMaxUnitNorm,
    makeVectorUnitNorm, makeVectorUnitNorm_inPlace, peigvech, characteristic_subspace_h,
    characteristic_subspace, matrixsqrth, orthogonal_complement, vec2diagMat,
    parallel_projection, orthogonal_projection, oblique_projection, regularize,
    zero2identity, mytorch_eigvalsh, mytorch_eigh,
    randdir, randdir_orthogonal2vec, sample_complex_multivariate, gaussian, wmean,
    norm_by_sum, deviation,
    windowing, windowing_conv, exp_windowing, exp_windowing_conv,
    exp_windowing_recursive, exp_windowing_recursive_changing_factor
)
from .tensor_ops import *
from .system import *
from .dsp import (
    calculate_t60, circularPositions, convolve_clean2microphone, convolve_white2microphone,
    rir2rtf, save_rirNoise2wav, simDiffuseNoise, simDiffuseNoiseANF, simRIR_shoebox, simRIR_shoebox_PRA,
    apply_filter_and_sum, Beamformer,
    plot_phaseogram, plot_spatial_coherence, plot_spectrogram, plot_stft_signal,
    getPower, getPower_dB, getPowerMat, snr2power,
    coherenceMatrix, gmsc, smoothCovarianceMatrix, smoothCovarianceMatrix_conv,
    wdo, windowedCovarianceMatrix,
    Frequency_Weighting, STFTtransform, slice2frames,
    GerkmannSPP, vad_g_from_SNR, vad_oracle, vad_oracle_batch, vad_opt_fast_gen,
    noise_whitening_robust, noise_whitening, noise_subtraction
)
from .metrics4torch import *
from .data_utils import *
from .model_utils import *

# If meta2df and save_audio were moved here from metrics:
try:
    from .meta2df import *
    from .save_audio import *
except ImportError:
    pass
