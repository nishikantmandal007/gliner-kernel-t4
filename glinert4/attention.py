import math

import torch

from .ops import ops
from .utils.constants import HEAD_SIZE

# DeBERTa scales c2c + c2p + p2c by sqrt(3 * head size); the kernel works in base 2
SCALE_LOG2 = (HEAD_SIZE * 3) ** -0.5 * math.log2(math.e)
PLAN_ALIGN_TOKENS = 32

radius_cache = {}
stock_forwards = {}


def attention(query, key, value, pos_key, pos_query, delta_to_local, position_offset, mask, lengths=None, radius=-1,
              cu_seqlens=None, max_seqlen=0):
    return ops().disentangled_attention(query, key, value, pos_key, pos_query, delta_to_local, position_offset, mask,
                                        SCALE_LOG2, lengths, radius, cu_seqlens, max_seqlen)


def real_lengths(mask):
    positions = torch.arange(1, mask.shape[1] + 1, device=mask.device, dtype=torch.int32)
    return (mask.to(torch.int32) * positions).amax(dim=1).to(torch.int32).contiguous()


# how far the relative-position table stays linear around the centre, so the kernel can compute those buckets
def linear_radius(plan):
    key = (plan.delta_to_local.data_ptr(), plan.position_offset)
    if key not in radius_cache:
        table, offset = plan.delta_to_local.cpu(), plan.position_offset
        radius = 0
        while (offset - radius - 1 >= 0 and offset + radius + 1 < table.numel()
               and int(table[offset + radius + 1]) - int(table[offset]) == radius + 1
               and int(table[offset]) - int(table[offset - radius - 1]) == radius + 1):
            radius += 1
        radius_cache[key] = radius
    return radius_cache[key]


def is_supported(module, hidden_states, plan):
    relative = module.relative_attention and {"c2p", "p2c"} <= set(module.pos_att_type)
    return (module.backend != "torch" and hidden_states.dtype == torch.float16
            and module.attention_head_size == HEAD_SIZE and relative and plan.pos_key is not None
            and plan.pos_query is not None)


def install():
    from disentangled_flash import kernel
    from disentangled_flash.packed import resolve_packed_info

    cls = kernel.InferenceDisentangledSelfAttention
    if cls in stock_forwards:
        return
    stock_prepared, stock_packed = cls.forward_prepared, cls.forward_packed

    def forward_prepared(self, hidden_states, attention_mask, plan):
        if not is_supported(self, hidden_states, plan):
            return stock_prepared(self, hidden_states, attention_mask, plan)
        batch_size, sequence_length = hidden_states.shape[:2]
        query, key, value = self._project_qkv(hidden_states)
        mask = None if self.assume_unpadded else attention_mask.to(torch.uint8).contiguous()
        output = attention(self._reshape_heads(query, batch_size, sequence_length),
                           self._reshape_heads(key, batch_size, sequence_length),
                           self._reshape_heads(value, batch_size, sequence_length),
                           plan.pos_key, plan.pos_query, plan.delta_to_local, plan.position_offset, mask,
                           lengths=real_lengths(mask) if mask is not None else None, radius=linear_radius(plan))
        return output, None

    def forward_packed(self, hidden_states, cu_seqlens, max_seqlen=None, *, rel_embeddings=None, packed_info=None):
        if self.backend == "torch":
            return stock_packed(self, hidden_states, cu_seqlens, max_seqlen, rel_embeddings=rel_embeddings,
                                packed_info=packed_info)
        info = resolve_packed_info(cu_seqlens, hidden_states.size(0), max_seqlen, packed_info)
        if self._cached_qkv_weight is None or (
                self._cached_pos_key is None and self._cached_pos_query is None
                and self._get_cached_shape_plan(info.max_seqlen, hidden_states.device) is None):
            self.prepare_for_inference(rel_embeddings)
        plan = self.prepare_shape(-(-info.max_seqlen // PLAN_ALIGN_TOKENS) * PLAN_ALIGN_TOKENS, hidden_states.device)
        if not is_supported(self, hidden_states, plan):
            return stock_packed(self, hidden_states, cu_seqlens, max_seqlen, rel_embeddings=rel_embeddings,
                                packed_info=packed_info)
        total = hidden_states.size(0)
        query, key, value = (tensor.view(total, self.num_attention_heads, HEAD_SIZE).permute(1, 0, 2)
                             for tensor in self._project_qkv(hidden_states))
        output = attention(query, key, value, plan.pos_key, plan.pos_query, plan.delta_to_local, plan.position_offset,
                           None, radius=linear_radius(plan), cu_seqlens=cu_seqlens.to(torch.int32).contiguous(),
                           max_seqlen=info.max_seqlen)
        return output, None

    cls.forward_prepared, cls.forward_packed = forward_prepared, forward_packed
    stock_forwards[cls] = (stock_prepared, stock_packed)


def uninstall():
    for cls, (prepared, packed) in stock_forwards.items():
        cls.forward_prepared, cls.forward_packed = prepared, packed
    stock_forwards.clear()
