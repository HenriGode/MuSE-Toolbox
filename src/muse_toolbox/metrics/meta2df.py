import torch
import pandas as pd
from muse_toolbox.metrics.base_metric import BaseMetric
from muse_toolbox.utils import STFTtransform


class META2DF(BaseMetric):
    """Metadata to DataFrame metric class.
    
    Inherits from BaseMetric. Instead of computing standard signal metrics,
    this class extracts scenario parameters (like SNR, segment lengths) from
    the metadata and compiles them into the evaluation dataframe.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    requires_reference = True

    input_snr: list[float]
    fixed_seg_length: list[float]
    scenario_ids: list[str]

    def __init__(self, transform: STFTtransform, model_name: str, *args, **kwargs):
        """Initializes the META2DF metric.

        Args:
            transform (STFTtransform): Transformer to convert STFT back to time-domain.
            model_name (str): Name of the model being evaluated.
            *args: Variable length arguments passed to BaseMetric.
            **kwargs: Arbitrary keyword arguments passed to BaseMetric.
        """
        super().__init__(*args, requires_numpy=False, **kwargs)

        self.transform = transform
        self.model_name = model_name

        self.ref_channel = 0  # Assuming the first channel is the reference TODO: make this configurable

        # add input snr and fixed seg length so that they are saved to the combined dataframe
        self.add_state("input_snr", default=[], dist_reduce_fx="cat")
        self.add_state("fixed_seg_length", default=[], dist_reduce_fx="cat")
        self.add_state("scenario_ids", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: list[
            tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]
        ],
        targets: tuple[dict, torch.Tensor],
        meta: dict,
        dataloader_idx: int,
    ):
        """Extracts and stores scenario parameters from batch metadata.

        Args:
            preds: List containing prediction dictionaries and outputs.
            targets (tuple[dict, torch.Tensor]): Tuple with ground truth references.
            meta (dict): Dictionary with scenario metadata like scenario_params.
            dataloader_idx (int): Current dataloader index.
        """
        for bidx in range(len(preds)):
            scenario_params = meta["scenario_params"][bidx]

            self.input_snr.append(scenario_params["snr"])
            self.fixed_seg_length.append(scenario_params["fixed_time_between_events"])

            self.scenario_ids.append(meta["scenario_id"][bidx])

    def compute(self) -> dict | None:
        """Returns a dummy result indicating that metadata was processed.

        Returns:
            dict | None: Dictionary with `added_meta` flag.
        """
        return {"added_meta": torch.tensor(1)}

    def get_dataframe(self) -> pd.DataFrame | None:
        """Constructs a DataFrame summarizing extracted scenario metadata.

        Returns:
            pd.DataFrame | None: Dataframe containing scenario parameters.
        """
        results_dict = {
            "input_snr": self.input_snr,
            "fixed_seg_length": self.fixed_seg_length,
        }
        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df
