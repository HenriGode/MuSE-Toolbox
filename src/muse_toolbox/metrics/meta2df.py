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
            "recording_id",
            "num_mics",
            "mic_indices",
            "num_frames",
            "num_sources",
            "signal_length",
            "snr",
            "sirs",
            "room_dims",
            "rt60",
            "mic_array",
            "mic_pos",
            "source_positions",
            "noise_file_path",
            "fixed_time_between_events",
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
            preds: List containing prediction tensors per item.
            targets (tuple[dict, torch.Tensor]): Ground truth references.
            meta (dict): Dictionary with scenario metadata like scenario_params.
            dataloader_idx (int): Current dataloader index.
        """
        _ = targets, dataloader_idx
        sp_list = meta.get("scenario_params", [])
        rec_ids = meta.get("recording_id", [])
        top_num_mics = meta.get("num_mics", [])
        top_mic_indices = meta.get("mic_indices", [])
        scenario_id_list = meta.get("scenario_id", [])

        for bidx in range(len(preds)):
            raw_sp = sp_list[bidx] if bidx < len(sp_list) else {}
            scenario_params = raw_sp if isinstance(raw_sp, dict) else {}

            for key in self.meta_keys:
                state_key = "scenario_transform" if key == "transform" else key
                if key == "num_frames":
                    pred_item = preds[bidx]
                    val = int(pred_item.shape[0]) if hasattr(pred_item, "shape") else 0
                else:
                    val = scenario_params.get(key, None)
                    if val is None and key == "recording_id" and bidx < len(rec_ids):
                        val = rec_ids[bidx]
                    elif val is None and key == "num_mics" and bidx < len(top_num_mics):
                        val = top_num_mics[bidx]
                    elif val is None and key == "mic_indices" and bidx < len(top_mic_indices):
                        val = top_mic_indices[bidx]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                getattr(self, state_key).append(val)

            sid = (
                scenario_id_list[bidx]
                if bidx < len(scenario_id_list)
                else f"item_{bidx}"
            )
            self.scenario_ids.append(sid)

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
