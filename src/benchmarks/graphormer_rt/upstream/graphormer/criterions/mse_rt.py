"""Mean squared error criterion for Graphormer-RT retention-time prediction."""

from __future__ import annotations

import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass.configs import FairseqDataclass


@register_criterion("mse_rt", dataclass=FairseqDataclass)
class GraphormerRTMSECriterion(FairseqCriterion):
    """Train the RT prediction branch directly with MSE.

    The vendored RP model returns two tensors: the first is the retention-time
    prediction branch and the second is the auxiliary uncertainty-like branch
    used by the original Gaussian loss. This criterion intentionally optimizes
    only the first branch, matching what the evaluator writes as Predicted RT.
    """

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]
        values = model(**sample["net_input"])
        prediction = values[0] if isinstance(values, tuple) else values

        prediction = prediction.squeeze()
        target = sample["target"].squeeze()
        if prediction.ndim == 0:
            prediction = prediction.unsqueeze(0)
        if target.ndim == 0:
            target = target.unsqueeze(0)

        loss = F.mse_loss(prediction.float(), target.float(), reduction="sum")
        logging_output = {
            "loss": loss.detach(),
            "sample_size": int(target.numel()),
            "ntokens": int(target.numel()),
            "nsentences": sample_size,
        }
        return loss, target.numel(), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=8)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
