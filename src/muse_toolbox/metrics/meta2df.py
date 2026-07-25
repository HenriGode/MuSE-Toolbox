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

        # self.meta_keys = [
        #     "num_sources", "signal_length", "snr", "sirs", "room_dims", 
        #     "rt60", "mic_array", "mic_pos", "source_positions", 
        #     "noise_file_path", "fixed_time_between_events", 
        #     "activity_pattern", "sources", "transform", "generator_id"
        # ]
        self.meta_keys = [
            "num_sources", "signal_length", "snr", "sirs", "room_dims", 
            "rt60", "mic_array", "mic_pos", "source_positions", 
            "noise_file_path", "fixed_time_between_events", 
        ]

        for key in self.meta_keys:
            # Avoid overwriting self.transform
            state_key = "scenario_transform" if key == "transform" else key
            self.add_state(state_key, default=[], dist_reduce_fx="cat")

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

            for key in self.meta_keys:
                state_key = "scenario_transform" if key == "transform" else key
                # Convert complex nested arrays/lists to lists for dataframe compatibility
                val = scenario_params.get(key, None)
                if hasattr(val, "tolist"):
                    val = val.tolist()
                getattr(self, state_key).append(val)

            self.scenario_ids.append(meta["scenario_id"][bidx])

    def compute(self) -> dict | None:
        """Returns a dummy result indicating that metadata was processed.

        Returns:
            dict | None: Dictionary with `added_meta` flag.
        """
        return {"added_meta": torch.tensor(1.0)}

    def get_dataframe(self) -> pd.DataFrame | None:
        """Constructs a DataFrame summarizing extracted scenario metadata.

        Returns:
            pd.DataFrame | None: Dataframe containing scenario parameters.
        """
        results_dict = {}
        for key in self.meta_keys:
            state_key = "scenario_transform" if key == "transform" else key
            results_dict[key] = getattr(self, state_key)

        df = pd.DataFrame(results_dict, index=self.scenario_ids)
        df.index.name = "scenario_id"
        return df
