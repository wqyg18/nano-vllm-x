import torch
from torch import nn
import triton
import triton.language as tl
import os

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

try:
    import flashinfer
except ImportError:
    flashinfer = None


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        requested_backend = os.getenv("NANOVLLM_ATTENTION_BACKEND", "auto")
        self.backend = self._resolve_backend(requested_backend)

    def _resolve_backend(self, requested_backend: str) -> str:
        if requested_backend not in ("auto", "flash-attn", "flashinfer"):
            raise ValueError(f"Unsupported attention backend: {requested_backend}")
        if requested_backend == "flashinfer":
            if flashinfer is None:
                raise RuntimeError("NANOVLLM_ATTENTION_BACKEND=flashinfer but flashinfer is not installed")
            return "flashinfer"
        if requested_backend == "flash-attn":
            return "flash-attn"
        if flashinfer is not None:
            return "flashinfer"
        return "flash-attn"

    def _build_seq_kv_from_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, seq_len: int):
        if seq_len == 0:
            return k_cache[:0], v_cache[:0]
        block_size = k_cache.size(1)
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_table[:num_blocks].to(torch.long)
        k = k_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        v = v_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        return k, v

    def _flashinfer_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        outputs = []
        num_seqs = context.cu_seqlens_q.numel() - 1
        for i in range(num_seqs):
            q_start, q_end = context.cu_seqlens_q[i].item(), context.cu_seqlens_q[i + 1].item()
            k_start, k_end = context.cu_seqlens_k[i].item(), context.cu_seqlens_k[i + 1].item()
            q_i = q[q_start:q_end]
            if context.block_tables is None:
                k_i = k[k_start:k_end]
                v_i = v[k_start:k_end]
            else:
                k_i, v_i = self._build_seq_kv_from_cache(k_cache, v_cache, context.block_tables[i], k_end - k_start)
            o_i = flashinfer.prefill.single_prefill_with_kv_cache(
                q_i,
                k_i,
                v_i,
                causal=True,
                sm_scale=self.scale,
            )
            outputs.append(o_i)
        return torch.cat(outputs, dim=0)

    def _flashinfer_decode(self, q: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        outputs = []
        for i, seq_len in enumerate(context.context_lens.tolist()):
            k_i, v_i = self._build_seq_kv_from_cache(k_cache, v_cache, context.block_tables[i], seq_len)
            o_i = flashinfer.decode.single_decode_with_kv_cache(
                q[i],
                k_i,
                v_i,
                sm_scale=self.scale,
            )
            outputs.append(o_i)
        return torch.stack(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if self.backend == "flashinfer":
            if context.is_prefill:
                o = self._flashinfer_prefill(q, k, v, context, k_cache, v_cache)
            else:
                o = self._flashinfer_decode(q, context, k_cache, v_cache)
            return o

        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:  # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o
