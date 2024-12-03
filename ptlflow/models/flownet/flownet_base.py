from ..base_model.base_model import BaseModel
from .losses import MultiScale


class FlowNetBase(BaseModel):
    def __init__(
        self,
        div_flow: float = 20.0,
        input_channels: int = 6,
        batch_norm: bool = False,
        loss_start_scale: int = 4,
        loss_num_scales: int = 5,
        loss_base_weight: float = 0.32,
        loss_norm: str = "L2",
        **kwargs,
    ):
        super(FlowNetBase, self).__init__(
            loss_fn=MultiScale(
                startScale=loss_start_scale,
                numScales=loss_num_scales,
                l_weight=loss_base_weight,
                norm=loss_norm,
            ),
            output_stride=64,
            **kwargs,
        )
        self.div_flow = div_flow
        self.input_channels = input_channels
        self.batch_norm = batch_norm
