import types
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:  # Transformers 4.43 compatibility
    from typing import TypedDict

    class FlashAttentionKwargs(TypedDict, total=False):
        pass

try:
    from transformers.processing_utils import Unpack
except ImportError:  # Python < 3.12 / Transformers 4.43 compatibility
    from typing_extensions import Unpack

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:  # Eager attention is used by all released experiments.
    ALL_ATTENTION_FUNCTIONS = {}
from transformers.cache_utils import Cache, DynamicCache
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "enable_evosparse",
    "enable_evosparse_without_cross_layer",
    "enable_evosparse_without_cross_step",
    "reset_evosparse_state",
]

neg_inf = torch.finfo(torch.float16).min

def update_heatv12_pure_attention(
    heat_cache: torch.Tensor,
    heat: torch.Tensor,
    selected_indices: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Decay accumulated importance and add scores for selected tokens."""
    # device = heat_cache.device
    decayed_heat = heat_cache * alpha

    new_heat_cache = F.pad(decayed_heat, (0, 1), value=0.0)

    new_heat_cache.scatter_add_(dim=0, index=selected_indices, src=heat)

    return new_heat_cache

def _get_baseline_indices(seq_len, sink_token, local_token, device):
    """Return indices for attention sinks and the local window."""
    sink_indices = torch.arange(min(sink_token, seq_len), device=device)
    local_start_index = max(0, seq_len - local_token)
    local_indices = torch.arange(local_start_index, seq_len, device=device)
    return torch.cat([sink_indices, local_indices])

def _get_heat_group_indicesv10(heat_cache, num_groups_to_select, group_size, sink_token, local_token, seq_len, device, exclude_groups: Optional[torch.Tensor] = None):
    """Select query-agnostic blocks using accumulated importance."""
    heat_range_start = sink_token
    heat_range_end = seq_len - local_token
    heat_range_len = heat_range_end - heat_range_start

    if heat_range_len <= 0 or num_groups_to_select <= 0:
        return torch.tensor([], device=device, dtype=torch.long)

    valid_heat = heat_cache[heat_range_start:heat_range_end]

    pad_len = (group_size - len(valid_heat) % group_size) % group_size
    heat_padded = F.pad(valid_heat, (0, pad_len), value=-1)
    num_groups = len(heat_padded) // group_size

    if num_groups == 0:
        return torch.tensor([], device=device, dtype=torch.long)

    heat_grouped = heat_padded.view(num_groups, group_size)

    group_scores = heat_grouped.sum(dim=-1)

    if exclude_groups is not None and len(exclude_groups) > 0:
        group_scores[exclude_groups] = float('-inf')

    k = min(num_groups_to_select, num_groups)

    num_available_groups = (group_scores > float('-inf')).sum()
    k = min(k, num_available_groups)

    if k <= 0:
        return torch.tensor([], device=device, dtype=torch.long)

    _ , top_groups = torch.topk(group_scores, k=k)
    return top_groups

def _get_retrieval_group_indices(query, key_states, scaling, retrieval_heads_for_layer, k_groups_per_head, group_size, sink_token, local_token, select_group_budget):
    """Select query-aware blocks using the configured retrieval heads."""
    device = query.device
    seq_len = key_states.shape[-2]

    select_query = query[:,retrieval_heads_for_layer,:,:]
    select_key_states = key_states[:,retrieval_heads_for_layer,:,:]

    select_attn_weight = torch.matmul(select_query,select_key_states.transpose(-2,-1)) * scaling

    num_retrieval_head = len(retrieval_heads_for_layer)

    heat_range_start = sink_token
    heat_range_end = seq_len - local_token
    heat_range_len = heat_range_end - heat_range_start

    if heat_range_len <= 0:
        return torch.tensor([], device=device, dtype=torch.long)

    all_top_groups = []
    for i in range(num_retrieval_head):
        heat_range_attn = select_attn_weight[0, i, 0, heat_range_start:heat_range_end]

        pad_len = (group_size - len(heat_range_attn) % group_size) % group_size
        padded_attn = F.pad(heat_range_attn, (0, pad_len), value=neg_inf)#0
        num_groups = len(padded_attn) // group_size

        if num_groups == 0:
            continue

        grouped_attn = padded_attn.view(num_groups, group_size)
        group_scores, _ = torch.max(grouped_attn, dim=-1)

        k = min(k_groups_per_head, num_groups)
        _ , top_groups = torch.topk(group_scores, k=k)
        all_top_groups.append(top_groups)

    if not all_top_groups:
        return torch.tensor([], device=device, dtype=torch.long)

    unique_group_indices = torch.unique(torch.cat(all_top_groups))

    if len(unique_group_indices) > select_group_budget:
        unique_list = list(dict.fromkeys(torch.cat(all_top_groups).tolist()))
        unique_group_indices = torch.tensor(unique_list, dtype=torch.long, device=device)
        unique_group_indices = unique_group_indices[:select_group_budget]

    return unique_group_indices

def _expand_group_indices_to_tokens(group_indices, group_size, sink_token, local_token, seq_len, device):
    """Expand block indices into global token indices."""
    if group_indices is None or len(group_indices) == 0:
        return torch.tensor([], device=device, dtype=torch.long)

    relative_indices = torch.arange(group_size, device=device)
    start_offsets = group_indices.unsqueeze(1) * group_size
    token_indices_in_heat_range = start_offsets + relative_indices
    global_indices = token_indices_in_heat_range.flatten() + sink_token

    valid_mask = global_indices < (seq_len - local_token)
    return global_indices[valid_mask]

def retrieval_heat_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_id,
    heat_cache,
    inherited_indices,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    kv_seq_len = key_states.shape[-2]
    device = key_states.device
    decay = module.decay
    retrieval_indices = None

    if attention_mask is not None:

        attention_mask = attention_mask.to(device)
        if kwargs['is_prefilling'] or (layer_id<2 and not module.enable_first_two_layers):  # pre-filling or first two layers
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            retrieval_indices = inherited_indices

        else:  # decoding
            retrieval_heads_in_layer = [head_idx for layer_idx, head_idx in kwargs['block_list'] if layer_idx == layer_id]
            is_retrieval_layer = len(retrieval_heads_in_layer) > 0

            group_size = module.group_size
            sink_token = module.sink_token
            local_token = module.local_token
            total_budget = module.total_budget
            k_groups_per_retrieval_head = module.per_head_top_group
            select_group_budget = (total_budget - sink_token - local_token)//group_size

            baseline_indices = _get_baseline_indices(kv_seq_len, sink_token, local_token, device)

            if inherited_indices is not None:
                inherited_indices = inherited_indices.to(device)

            if is_retrieval_layer:
                own_group_indices = _get_retrieval_group_indices(
                    query, key_states, scaling, retrieval_heads_in_layer, k_groups_per_retrieval_head,
                    group_size, sink_token, local_token, select_group_budget
                )
                if inherited_indices is not None:
                    result_retrieval_group_budget = select_group_budget - len(own_group_indices)
                    retrieval_indices = torch.unique(torch.cat([own_group_indices, inherited_indices[:result_retrieval_group_budget]]))
                else:
                    retrieval_indices = own_group_indices

            elif inherited_indices is not None:
                retrieval_indices = inherited_indices
            else:
                retrieval_indices = inherited_indices

            if retrieval_indices is not None:
                retrieval_group_num = len(retrieval_indices)
            else:
                retrieval_group_num = 0

            num_heat_groups = total_budget//group_size - retrieval_group_num - sink_token//group_size - local_token//group_size
            sparse_group_indices = _get_heat_group_indicesv10(heat_cache, num_heat_groups, group_size, sink_token, local_token, kv_seq_len, device, retrieval_indices)

            if retrieval_indices is not None:
                final_group_indices = torch.unique(torch.cat([retrieval_indices, sparse_group_indices]))
            else:
                final_group_indices = sparse_group_indices

            sparse_token_indices = _expand_group_indices_to_tokens(
                final_group_indices, group_size, sink_token, local_token, kv_seq_len, device
            )

            allowed_indices = torch.unique(torch.cat([baseline_indices, sparse_token_indices]))

            selected_key_states = key_states[:,:,allowed_indices,:]
            selected_value_states = value_states[:,:,allowed_indices,:]

            attn_weights = torch.matmul(query, selected_key_states.transpose(2, 3)) * scaling
            attn_weights_new = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if kwargs['is_prefilling'] or (layer_id<2 and not module.enable_first_two_layers):  # pre-filling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        batch, head, q_len, k_len = attn_weights.shape
        result = attn_weights.clone()
        if heat_cache.shape[0]!=0:
            heat_cache = F.pad(heat_cache, (0, q_len), value=0.0)
            result[:, :, 0, :] = result[:, :, 0, :] + heat_cache * decay

        for i in range(1, q_len):
            result[:, :, i, :] = result[:, :, i, :] + result[:, :, i-1, :] * decay

        heat = result[:,:,-1,:].mean(dim=1)[0]

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    else: #decoding
        heat_per_head = attn_weights_new  # [B, Head, Q_len, K_len]
        heat = heat_per_head.mean(dim=1)[
            0, 0
        ]  # [1, 1, K_len] Decoding Stage: Batch == 1; Query == 1
        heat = update_heatv12_pure_attention(heat_cache.to(heat.device), heat,selected_indices=allowed_indices,alpha=decay)
        attn_output = torch.matmul(attn_weights_new, selected_value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, heat, retrieval_indices

def heat_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_id,
    heat_cache,
    inherited_indices,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    kv_seq_len = key_states.shape[-2]
    device = key_states.device

    retrieval_indices = None
    if attention_mask is not None:

        attention_mask = attention_mask.to(device)
        if kwargs['is_prefilling'] or (layer_id<2 and not module.enable_first_two_layers):  # pre-filling or first two layers
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            retrieval_indices = inherited_indices

        else:  # decoding
            retrieval_heads_in_layer = [head_idx for layer_idx, head_idx in kwargs['block_list'] if layer_idx == layer_id]
            is_retrieval_layer = len(retrieval_heads_in_layer) > 0

            group_size = module.group_size
            sink_token = module.sink_token
            local_token = module.local_token
            total_budget = module.total_budget
            k_groups_per_retrieval_head = module.per_head_top_group
            select_group_budget = (total_budget - sink_token - local_token)//group_size

            baseline_indices = _get_baseline_indices(kv_seq_len, sink_token, local_token, device)

            num_heat_groups = total_budget//group_size  - sink_token//group_size - local_token//group_size
            sparse_group_indices = _get_heat_group_indicesv10(heat_cache, num_heat_groups, group_size, sink_token, local_token, kv_seq_len, device, retrieval_indices)

            if retrieval_indices is not None:
                final_group_indices = torch.unique(torch.cat([retrieval_indices, sparse_group_indices]))
            else:
                final_group_indices = sparse_group_indices

            sparse_token_indices = _expand_group_indices_to_tokens(
                final_group_indices, group_size, sink_token, local_token, kv_seq_len, device
            )

            allowed_indices = torch.unique(torch.cat([baseline_indices, sparse_token_indices]))

            selected_key_states = key_states[:,:,allowed_indices,:]
            selected_value_states = value_states[:,:,allowed_indices,:]

            attn_weights = torch.matmul(query, selected_key_states.transpose(2, 3)) * scaling
            attn_weights_new = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if kwargs['is_prefilling'] or (layer_id<2 and not module.enable_first_two_layers):  # pre-filling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        batch, head, q_len, k_len = attn_weights.shape

        result = attn_weights.clone()
        if heat_cache.shape[0]!=0:
            heat_cache = F.pad(heat_cache, (0, q_len), value=0.0)
            result[:, :, 0, :] = result[:, :, 0, :] + heat_cache * 0.996

        for i in range(1, q_len):
            result[:, :, i, :] = result[:, :, i, :] + result[:, :, i-1, :] * 0.996

        heat = result[:,:,-1,:].mean(dim=1)[0]

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    else: #decoding
        heat_per_head = attn_weights_new  # [B, Head, Q_len, K_len]
        heat = heat_per_head.mean(dim=1)[
            0, 0
        ]  # [1, 1, K_len] Decoding Stage: Batch == 1; Query == 1
        heat = update_heatv12_pure_attention(heat_cache.to(heat.device), heat,selected_indices=allowed_indices,alpha=0.996)
        attn_output = torch.matmul(attn_weights_new, selected_value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, heat, retrieval_indices

def retrieval_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_id,
    heat_cache,
    inherited_indices,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    kv_seq_len = key_states.shape[-2]
    device = key_states.device
    decay = module.decay
    retrieval_indices = None
    if attention_mask is not None:

        attention_mask = attention_mask.to(device)
        if kwargs['is_prefilling'] or (layer_id<2 and not module.enable_first_two_layers):  # pre-filling or first two layers
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            retrieval_indices = inherited_indices

        else:  # decoding
            retrieval_heads_in_layer = [head_idx for layer_idx, head_idx in kwargs['block_list'] if layer_idx == layer_id]
            is_retrieval_layer = len(retrieval_heads_in_layer) > 0

            group_size = module.group_size
            sink_token = module.sink_token
            local_token = module.local_token
            total_budget = module.total_budget
            k_groups_per_retrieval_head = module.per_head_top_group
            select_group_budget = (total_budget - sink_token - local_token)//group_size

            baseline_indices = _get_baseline_indices(kv_seq_len, sink_token, local_token, device)

            if inherited_indices is not None:
                inherited_indices = inherited_indices.to(device)

            if is_retrieval_layer:
                own_group_indices = _get_retrieval_group_indices(
                    query, key_states, scaling, retrieval_heads_in_layer, k_groups_per_retrieval_head,
                    group_size, sink_token, local_token, select_group_budget
                )
                if inherited_indices is not None:
                    result_retrieval_group_budget = select_group_budget - len(own_group_indices)
                    retrieval_indices = torch.unique(torch.cat([own_group_indices, inherited_indices[:result_retrieval_group_budget]]))
                else:
                    retrieval_indices = own_group_indices

            elif inherited_indices is not None:
                retrieval_indices = inherited_indices
            else:
                retrieval_indices = inherited_indices

            if retrieval_indices is not None:
                retrieval_group_num = len(retrieval_indices)
            else:
                retrieval_group_num = 0

            if retrieval_indices is not None:
                sparse_token_indices = _expand_group_indices_to_tokens(
                    retrieval_indices, group_size, sink_token, local_token, kv_seq_len, device
                )
                allowed_indices = torch.unique(torch.cat([baseline_indices, sparse_token_indices]))
            else:
                allowed_indices = baseline_indices


            selected_key_states = key_states[:,:,allowed_indices,:]
            selected_value_states = value_states[:,:,allowed_indices,:]

            attn_weights = torch.matmul(query, selected_key_states.transpose(2, 3)) * scaling
            attn_weights_new = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if kwargs['is_prefilling'] or (layer_id<2 and not module.enable_first_two_layers):  # pre-filling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        batch, head, q_len, k_len = attn_weights.shape

        result = attn_weights.clone()
        if heat_cache.shape[0]!=0:
            heat_cache = F.pad(heat_cache, (0, q_len), value=0.0)
            result[:, :, 0, :] = result[:, :, 0, :] + heat_cache * decay

        for i in range(1, q_len):
            result[:, :, i, :] = result[:, :, i, :] + result[:, :, i-1, :] * decay

        heat = result[:,:,-1,:].mean(dim=1)[0]

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    else: #decoding
        heat_per_head = attn_weights_new  # [B, Head, Q_len, K_len]
        heat = heat_per_head.mean(dim=1)[
            0, 0
        ]  # [1, 1, K_len] Decoding Stage: Batch == 1; Query == 1
        heat = update_heatv12_pure_attention(heat_cache.to(heat.device), heat,selected_indices=allowed_indices,alpha=decay)
        attn_output = torch.matmul(attn_weights_new, selected_value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, heat, retrieval_indices

def evosparse_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inherited_indices: Optional[torch.Tensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings

    cos = cos.to(query_states.device)
    sin = sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = retrieval_heat_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            #     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
            #     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            # )
            print(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, heat, retrieval_indices = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer_id=self.layer_id,
        heat_cache=self.heat,
        inherited_indices=inherited_indices,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.head_dim**-0.5,
        **kwargs,
    )
    self.heat = heat

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, retrieval_indices

def evosparse_without_cross_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inherited_indices: Optional[torch.Tensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings

    cos = cos.to(query_states.device)
    sin = sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = heat_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            #     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
            #     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            # )
            print(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, heat, retrieval_indices = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer_id=self.layer_id,
        heat_cache=self.heat,
        inherited_indices=inherited_indices,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.head_dim**-0.5,
        **kwargs,
    )
    self.heat = heat

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, retrieval_indices


def evosparse_without_cross_step_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inherited_indices: Optional[torch.Tensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings

    cos = cos.to(query_states.device)
    sin = sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = retrieval_forward

    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            #     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
            #     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            # )
            print(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, heat, retrieval_indices = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer_id=self.layer_id,
        heat_cache=self.heat,
        inherited_indices=inherited_indices,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.head_dim**-0.5,
        **kwargs,
    )
    self.heat = heat

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, retrieval_indices

def custom_llama_decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        inherited_indices: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_retrieval_indices = self.self_attn(
        # hidden_states, self_attn_weights, new_retrieval_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            inherited_indices=inherited_indices,
            **kwargs,
        )

        hidden_states = hidden_states.to(residual.device) #
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states.to(residual.device) #
        hidden_states = residual + hidden_states

        # outputs = (hidden_states,)
        outputs = (hidden_states, new_retrieval_indices)
        # if output_attentions:
        #     outputs += (self_attn_weights,)

        return outputs

def custom_llama_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)


        PROPAGATION_WINDOW_SIZE = 32

        propagated_indices = None
        propagation_lifespan = 0

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if propagation_lifespan > 0:
                indices_to_pass = propagated_indices
                propagation_lifespan -= 1
            else:
                indices_to_pass = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                inherited_indices=indices_to_pass,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]
            new_retrieval_indices = layer_outputs[1]

            if new_retrieval_indices is not None:
                propagated_indices = new_retrieval_indices
                propagation_lifespan = PROPAGATION_WINDOW_SIZE

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def enable_evosparse(model: LlamaForCausalLM, group_size=16, total_budget=4096, sink_token=16, local_token=128, per_head_top_group=30, retrieval_head_num=10, enable_first_two_layers=False, decay=0.996):
    """
    Enable EvoSparse on a Hugging Face ``LlamaForCausalLM`` model.
    """
    model.model.original_forward = model.model.forward
    model.model.forward = types.MethodType(custom_llama_model_forward, model.model)
    print("Patched LlamaModel.forward")

    for i, layer in enumerate(model.model.layers):
        layer.original_forward = layer.forward
        layer.forward = types.MethodType(custom_llama_decoder_layer_forward, layer)

        attn_module = layer.self_attn
        attn_module.enable_first_two_layers = enable_first_two_layers
        attn_module.layer_id = i
        attn_module.heat = torch.tensor([], dtype=torch.bfloat16)

        attn_module.group_size = group_size
        attn_module.total_budget = total_budget
        attn_module.sink_token = sink_token
        attn_module.local_token = local_token
        attn_module.per_head_top_group = per_head_top_group
        attn_module.retrieval_head_num = retrieval_head_num
        attn_module.decay = decay

        attn_module.original_forward = attn_module.forward
        attn_module.forward = types.MethodType(evosparse_attention_forward, attn_module)

    print(f"EvoSparse enabled for {len(model.model.layers)} Llama layers.")

def enable_evosparse_without_cross_layer(model: LlamaForCausalLM, group_size=16, total_budget=4096, sink_token=16, local_token=128, per_head_top_group=30, retrieval_head_num=10, enable_first_two_layers=False, decay=0.996):
    """
    Enable the EvoSparse ablation without Cross-Layer Propagation.
    """
    model.model.original_forward = model.model.forward
    model.model.forward = types.MethodType(custom_llama_model_forward, model.model)
    print("Patched LlamaModel.forward")

    for i, layer in enumerate(model.model.layers):
        layer.original_forward = layer.forward
        layer.forward = types.MethodType(custom_llama_decoder_layer_forward, layer)

        attn_module = layer.self_attn
        attn_module.enable_first_two_layers = enable_first_two_layers
        attn_module.layer_id = i
        attn_module.heat = torch.tensor([], dtype=torch.bfloat16)
        attn_module.current_position = 0

        attn_module.group_size = group_size
        attn_module.total_budget = total_budget
        attn_module.sink_token = sink_token
        attn_module.local_token = local_token
        attn_module.per_head_top_group = per_head_top_group
        attn_module.retrieval_head_num = retrieval_head_num
        attn_module.decay = decay

        attn_module.original_forward = attn_module.forward
        attn_module.forward = types.MethodType(evosparse_without_cross_layer_forward, attn_module)

    print(f"EvoSparse without Cross-Layer Propagation enabled for {len(model.model.layers)} Llama layers.")

def enable_evosparse_without_cross_step(model: LlamaForCausalLM, group_size=16, total_budget=4096, sink_token=16, local_token=128, per_head_top_group=30, retrieval_head_num=10, enable_first_two_layers=False, decay=0.996):
    """
    Enable the EvoSparse ablation without Cross-Step Accumulation.
    """
    model.model.original_forward = model.model.forward
    model.model.forward = types.MethodType(custom_llama_model_forward, model.model)
    print("Patched LlamaModel.forward")

    for i, layer in enumerate(model.model.layers):
        layer.original_forward = layer.forward
        layer.forward = types.MethodType(custom_llama_decoder_layer_forward, layer)

        attn_module = layer.self_attn
        attn_module.enable_first_two_layers = enable_first_two_layers
        attn_module.layer_id = i
        attn_module.heat = torch.tensor([], dtype=torch.bfloat16)
        attn_module.current_position = 0

        attn_module.group_size = group_size
        attn_module.total_budget = total_budget
        attn_module.sink_token = sink_token
        attn_module.local_token = local_token
        attn_module.per_head_top_group = per_head_top_group
        attn_module.retrieval_head_num = retrieval_head_num
        attn_module.decay = decay

        attn_module.original_forward = attn_module.forward
        attn_module.forward = types.MethodType(evosparse_without_cross_step_forward, attn_module)

    print(f"EvoSparse without Cross-Step Accumulation enabled for {len(model.model.layers)} Llama layers.")

def reset_evosparse_state(model: LlamaForCausalLM):
    for i, layer in enumerate(model.model.layers):
        attn_module = layer.self_attn
        attn_module.heat = torch.tensor([], dtype=torch.bfloat16)
