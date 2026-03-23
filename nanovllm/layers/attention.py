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

FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


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

    def _get_flashinfer_runtime(self, context, device: torch.device):
        runtime = getattr(context, "_flashinfer_runtime", None)
        if runtime is None:
            runtime = {}
            context._flashinfer_runtime = runtime
        device_key = str(device)
        if device_key not in runtime:
            runtime[device_key] = dict(
                workspace=torch.zeros(
                    FLASHINFER_WORKSPACE_BYTES,
                    dtype=torch.uint8,
                    device=device,
                ),
                decode_wrappers={},
                paged_prefill_wrappers={},
                ragged_prefill_wrappers={},
            )
        return runtime[device_key]

    def _seq_lens_from_cu_seqlens(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        return cu_seqlens[1:] - cu_seqlens[:-1]

    def _build_paged_kv_metadata(self, block_tables: torch.Tensor, seq_lens: torch.Tensor, block_size: int):
        batch_size = seq_lens.numel()
        seq_lens_list = seq_lens.tolist()
        indptr = [0]
        page_indices = []
        last_page_len = []

        for i, seq_len in enumerate(seq_lens_list):
            if seq_len <= 0:
                indptr.append(indptr[-1])
                last_page_len.append(0)
                continue
            num_blocks = (seq_len + block_size - 1) // block_size
            indptr.append(indptr[-1] + num_blocks)
            last_page_len.append((seq_len - 1) % block_size + 1)
            if num_blocks:
                page_indices.append(block_tables[i, :num_blocks])

        indices = (
            torch.cat(page_indices, dim=0)
            if page_indices
            else torch.empty(0, dtype=block_tables.dtype, device=block_tables.device)
        )
        return (
            torch.tensor(indptr, dtype=torch.int32, device=block_tables.device),
            indices.to(dtype=torch.int32),
            torch.tensor(last_page_len, dtype=torch.int32, device=block_tables.device).view(batch_size),
        )

    def _get_flashinfer_ragged_prefill_wrapper(self, q: torch.Tensor, k: torch.Tensor, context):
        runtime = self._get_flashinfer_runtime(context, q.device)
        key = (self.num_heads, self.num_kv_heads, self.head_dim, self.scale, q.dtype, k.dtype)
        wrapper = runtime["ragged_prefill_wrappers"].get(key)
        if wrapper is None:
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                runtime["workspace"],
                kv_layout="NHD",
            )
            wrapper.plan(
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                causal=True,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                seq_lens=self._seq_lens_from_cu_seqlens(context.cu_seqlens_k),
                seq_lens_q=self._seq_lens_from_cu_seqlens(context.cu_seqlens_q),
            )
            runtime["ragged_prefill_wrappers"][key] = wrapper
        return wrapper

    def _get_flashinfer_paged_prefill_wrapper(self, q: torch.Tensor, k_cache: torch.Tensor, context):
        runtime = self._get_flashinfer_runtime(context, q.device)
        block_size = k_cache.size(1)
        key = (self.num_heads, self.num_kv_heads, self.head_dim, self.scale, q.dtype, k_cache.dtype, block_size)
        wrapper = runtime["paged_prefill_wrappers"].get(key)
        if wrapper is None:
            seq_lens = self._seq_lens_from_cu_seqlens(context.cu_seqlens_k)
            seq_lens_q = self._seq_lens_from_cu_seqlens(context.cu_seqlens_q)
            kv_indptr, kv_indices, kv_last_page_len = self._build_paged_kv_metadata(
                context.block_tables,
                seq_lens,
                block_size,
            )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                runtime["workspace"],
                kv_layout="NHD",
            )
            wrapper.plan(
                context.cu_seqlens_q,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                block_size,
                causal=True,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k_cache.dtype,
                seq_lens=seq_lens,
                seq_lens_q=seq_lens_q,
                block_tables=context.block_tables,
            )
            runtime["paged_prefill_wrappers"][key] = wrapper
        return wrapper

    def _get_flashinfer_decode_wrapper(self, q: torch.Tensor, k_cache: torch.Tensor, context):
        runtime = self._get_flashinfer_runtime(context, q.device)
        block_size = k_cache.size(1)
        use_tensor_cores = self.num_heads != self.num_kv_heads
        key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.scale,
            q.dtype,
            k_cache.dtype,
            block_size,
            use_tensor_cores,
        )
        wrapper = runtime["decode_wrappers"].get(key)
        if wrapper is None:
            kv_indptr, kv_indices, kv_last_page_len = self._build_paged_kv_metadata(
                context.block_tables,
                context.context_lens,
                block_size,
            )
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                runtime["workspace"],
                kv_layout="NHD",
                use_tensor_cores=use_tensor_cores,
            )
            wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                block_size,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k_cache.dtype,
                block_tables=context.block_tables,
                seq_lens=context.context_lens,
            )
            runtime["decode_wrappers"][key] = wrapper
        return wrapper

    def _flashinfer_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        if context.block_tables is None:
            wrapper = self._get_flashinfer_ragged_prefill_wrapper(q, k, context)
            return wrapper.run(q, k, v)
        wrapper = self._get_flashinfer_paged_prefill_wrapper(q, k_cache, context)
        return wrapper.run(q, (k_cache, v_cache))

    def _flashinfer_decode(self, q: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        wrapper = self._get_flashinfer_decode_wrapper(q, k_cache, context)
        return wrapper.run(q, (k_cache, v_cache))

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
