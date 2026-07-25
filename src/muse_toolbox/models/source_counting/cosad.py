import logging
from typing import Any

from muse_toolbox.models.base_model import BaseLitModel
from muse_toolbox.models.components.feature_extractors import BaseFeatureExtractor
from muse_toolbox.models.source_counting.estimators import BaseSourceCountEstimator
from muse_toolbox.models.components.channel_combinator.base_channel_combinator import BaseChannelCombinator
from muse_toolbox.data.components.heterogeneous_batch import HeterogeneousBatch
from muse_toolbox.utils import STFTtransform

log = logging.getLogger(__name__)


class COSADmodule(BaseLitModel):
    """
    A PyTorch Lightning module for Continuous Online Source Activity Detection (COSAD).

    This module orchestrates a two-stage pipeline:
    1. Feature Extraction: Processes raw STFT input into discriminative features.
    2. Source Count Estimation: Uses the extracted features to estimate the number of 
       active sources over time.
    """

    def __init__(
        self,
        transform: STFTtransform,
        feature_extractor: BaseFeatureExtractor,
        source_count_estimator: BaseSourceCountEstimator,
        channel_combinator: BaseChannelCombinator | Any | None = None,
        batch_size: int = 1,
        loss_config: dict[str, Any] = {"CrossEntropy": None},
        optimizer_config: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
        metrics_train: dict[str, Any] | None = None,
        metrics_val: dict[str, Any] | None = None,
        metrics_test: dict[str, Any] | None = None,
        compute_complexity_metrics: bool = False,
        check_causality: bool = False,
        permute_channels: bool = True,
    ):
        """
        Initializes the COSADmodule.

        Args:
            transform (utilities.STFTtransform): The STFT configuration for transforming audio signals.
            feature_extractor (BaseFeatureExtractor): Module responsible for extracting features from the STFT.
            source_count_estimator (BaseSourceCountEstimator): Module responsible for predicting source activity.
            channel_combinator (Optional[BaseChannelCombinator]): Module for condensing channel features.
            batch_size (int): Batch size for processing.
            loss_config (dict[str, Any]): Configuration for the loss function.
            optimizer_config (Optional[dict[str, Any]]): Configuration for the optimizer.
            lr_scheduler_config (Optional[dict[str, Any]]): Configuration for the learning rate scheduler.
            metrics_train (Optional[dict[str, Any]]): Metrics to track during training.
            metrics_val (Optional[dict[str, Any]]): Metrics to track during validation.
            metrics_test (Optional[dict[str, Any]]): Metrics to track during testing.
            compute_complexity_metrics (bool): Whether to profile computational complexity.
            check_causality (bool): Whether to enforce causality checks on the model.
            permute_channels (bool): If True, randomly permutes channels during training.
        """
        import functools
        if isinstance(feature_extractor, functools.partial):
            feature_extractor = feature_extractor(transform=transform)
            
        if isinstance(channel_combinator, functools.partial):
            channel_combinator = channel_combinator(
                input_feature_dim=feature_extractor.feature_dim
            )
            
        estimator_input_dim = feature_extractor.feature_dim
        # If a channel combinator is present, it handles channel condensation but preserves feature dim
        # (or provides an explicit out_feature_dim if it changes it).
        if channel_combinator is not None:
            if hasattr(channel_combinator, 'out_feature_dim'):
                estimator_input_dim = channel_combinator.out_feature_dim

        if isinstance(source_count_estimator, functools.partial):
            source_count_estimator = source_count_estimator(
                input_dim=estimator_input_dim,
                transform=transform
            )
            
        super().__init__(
            model_name=f"COSAD_{feature_extractor.__class__.__name__}_{source_count_estimator.__class__.__name__}",
            batch_size=batch_size,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            metrics_test=metrics_test,
            compute_complexity_metrics=compute_complexity_metrics,
            check_causality=check_causality,
            transform=transform,
        )

        self.transform = transform

        # Assign injected dependencies
        self.feature_extractor = feature_extractor
        self.channel_combinator = channel_combinator
        self.source_count_estimator = source_count_estimator
        self.permute_channels = permute_channels

        self.num_params = self.count_parameters()

        # Save hyperparameters, but ignore the complex objects (modules)
        # as they are part of the model structure, not just config params.
        self.save_hyperparameters(
            ignore=["feature_extractor", "source_count_estimator", "channel_combinator", "transform"]
        )

        # Manually save the configs of the injected dependencies
        self.hparams["feature_config"] = self.feature_extractor.get_config()
        if self.channel_combinator is not None and hasattr(self.channel_combinator, "get_config"):
            self.hparams["combinator_config"] = self.channel_combinator.get_config()
        self.hparams["estimator_config"] = self.source_count_estimator.get_config()
        self.hparams["transform_config"] = self.transform.get_config()

        self._verbose_parameters()

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Logs the parameters of the module and its sub-components in a structured format.

        Args:
            indent (str): A string to prepend to each log line for indentation.
        """
        log.info(f"{indent}{self.__class__.__name__} Parameters:")
        log.info(f"{indent}  Batch Size: {self.batch_size}")
        log.info(f"{indent}  Test Metrics: {self.metrics_test}")
        
        if hasattr(self.transform, "_verbose_parameters"):
            self.transform._verbose_parameters(indent=indent + "  ")

        if hasattr(self.feature_extractor, "_verbose_parameters"):
            self.feature_extractor._verbose_parameters(indent=indent + "  ")
        else:
            log.info(
                f"{indent}  Feature Extractor: {self.feature_extractor.__class__.__name__}"
            )
            
        if self.channel_combinator is not None:
            if hasattr(self.channel_combinator, "_verbose_parameters"):
                self.channel_combinator._verbose_parameters(indent=indent + "  ")
            else:
                log.info(
                    f"{indent}  Channel Combinator: {self.channel_combinator.__class__.__name__}"
                )

        if hasattr(self.source_count_estimator, "_verbose_parameters"):
            self.source_count_estimator._verbose_parameters(indent=indent + "  ")
        else:
            log.info(
                f"{indent}  Source Count Estimator: {self.source_count_estimator.__class__.__name__}"
            )

    def forward_(self, batch: HeterogeneousBatch) -> HeterogeneousBatch:
        """
        Executes the forward pass of the COSAD pipeline.

        Applies the feature extractor followed by the source count estimator 
        to the provided batch data.

        Args:
            batch (HeterogeneousBatch): A batch object containing the STFT input 
                data and relevant metadata.

        Returns:
            HeterogeneousBatch: The processed batch, now populated with source 
                activity estimates.
        """

        # 0. Data Augmentation
        if self.training and self.permute_channels:
            batch.randomly_permute_channels()

        # 1. Feature Extraction
        batch.apply_feature_extractor(self.feature_extractor)
        
        # 2. Channel Combination (Optional)
        if self.channel_combinator is not None:
            batch.apply_channel_combinator(self.channel_combinator)

        # 3. Detection (Source Count Estimation)
        # The estimator takes the features and estimates source activity.
        batch.apply_source_count_estimator(self.source_count_estimator)
        return batch

    def predict_step(
        self, batch: HeterogeneousBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> HeterogeneousBatch:
        """
        Executes a single prediction step and performs argmax on the estimates.

        Args:
            batch (HeterogeneousBatch): The prediction batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader.

        Returns:
            HeterogeneousBatch: The batch with discretized (argmax) source count estimates.
        """
        predictions = super().predict_step(batch, batch_idx, dataloader_idx)
        predictions.estimates = [est.argmax(dim=-1) for est in predictions.estimates]
        return predictions
