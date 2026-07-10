import torch
import math
import matplotlib
from muse_toolbox.models.building_blocks.feature_extractors import BaseFeatureExtractor
from muse_toolbox.models.building_blocks.source_count_estimators import BaseSourceCountEstimator
from utilities.data_utils import HeterogeneousBatch
import muse_toolbox.utils as utilities
from muse_toolbox.models.common.base_model import BaseLitModel
from typing import Optional, Any

matplotlib.use("agg")

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class COSADmodule(BaseLitModel):

    def __init__(
        self,
        transform: utilities.STFTtransform,
        feature_extractor: BaseFeatureExtractor,
        source_count_estimator: BaseSourceCountEstimator,
        batch_size: int = 1,
        loss_config: dict = {"CrossEntropy": None},
        optimizer_config: Optional[dict] = None,
        lr_scheduler_config: Optional[dict] = None,
        metrics_train: Optional[dict] = None,
        metrics_val: Optional[dict] = None,
        metrics_test: Optional[dict] = None,
        compute_complexity_metrics: bool = False,
        check_causality: bool = False,
    ):
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
        self.source_count_estimator = source_count_estimator

        self.num_params = self.count_parameters()

        # Save hyperparameters, but ignore the complex objects (modules)
        # as they are part of the model structure, not just config params.
        self.save_hyperparameters(
            ignore=["feature_extractor", "source_count_estimator", "transform"]
        )

        # Manually save the configs of the injected dependencies
        self.hparams["feature_config"] = self.feature_extractor.get_config()
        self.hparams["estimator_config"] = self.source_count_estimator.get_config()
        self.hparams["transform_config"] = self.transform.get_config()

        self._verbose_parameters()

    def _verbose_parameters(self, indent: str = "") -> None:
        """
        Prints the parameters of the module in a structured, indented format.

        Args:
            indent (str, optional): A string to prepend to each line for indentation.
                                    Defaults to "".
        """
        print(f"{indent}{self.__class__.__name__} Parameters:")
        print(f"{indent}  Batch Size: {self.batch_size}")
        print(f"{indent}  Test Metrics: {self.metrics_test}")
        self.transform._verbose_parameters(indent=indent + "  ")

        if hasattr(self.feature_extractor, "_verbose_parameters"):
            self.feature_extractor._verbose_parameters(indent=indent + "  ")
        else:
            print(
                f"{indent}  Feature Extractor: {self.feature_extractor.__class__.__name__}"
            )

        if hasattr(self.source_count_estimator, "_verbose_parameters"):
            self.source_count_estimator._verbose_parameters(indent=indent + "  ")
        else:
            print(
                f"{indent}  Source Count Estimator: {self.source_count_estimator.__class__.__name__}"
            )

    def forward_(
        self,
        batch: HeterogeneousBatch,
    ) -> HeterogeneousBatch:
        """
        Forward pass of the LightningModule.

        The input `batch` is expected to be a dictionary (from the DataModule).
        It should contain the 'input_type' key so the feature extractor knows what to do.
        """

        # 1. Feature Extraction
        batch.apply_feature_extractor(self.feature_extractor)

        # 2. Detection (Source Count Estimation)
        # The estimator takes the features and estimates source activity.
        batch.apply_source_count_estimator(self.source_count_estimator)
        return batch

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        predictions = super().predict_step(*args, **kwargs)
        predictions.estimates = [est.argmax(dim=-1) for est in predictions.estimates]
        return predictions
