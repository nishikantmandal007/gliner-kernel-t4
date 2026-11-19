import torch

from .ops import ops
from .utils.constants import HIDDEN_SIZE

stock_forwards = {}


def install():
    from transformers.models.deberta_v2 import modeling_deberta_v2 as deberta

    for cls in (deberta.DebertaV2SelfOutput, deberta.DebertaV2Output):
        if cls in stock_forwards:
            continue
        stock = cls.forward

        def forward(self, hidden_states, input_tensor, stock_forward=stock):
            norm = self.LayerNorm
            if (self.training or hidden_states.dtype != torch.float16 or not hidden_states.is_cuda
                    or norm.normalized_shape != (HIDDEN_SIZE,) or norm.weight is None):
                return stock_forward(self, hidden_states, input_tensor)
            return ops().add_layernorm(self.dense(hidden_states).contiguous(), input_tensor.contiguous(), norm.weight,
                                       norm.bias, norm.eps)

        cls.forward = forward
        stock_forwards[cls] = stock


def uninstall():
    for cls, stock in stock_forwards.items():
        cls.forward = stock
    stock_forwards.clear()
