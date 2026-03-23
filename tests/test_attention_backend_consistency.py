import os
import unittest

import torch

from nanovllm.layers.attention import Attention, flashinfer
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for attention backend consistency tests")
@unittest.skipUnless(flashinfer is not None, "flashinfer is required for attention backend consistency tests")
class TestAttentionBackendConsistency(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        # Use model-like dimensions to avoid kernel corner-case limitations.
        self.num_heads = 8
        self.num_kv_heads = 8
        self.head_dim = 128
        self.scale = self.head_dim ** -0.5
        # fp16 kernels from different backends can have tiny numerical deltas.
        self.rtol = 5e-3
        self.atol = 5e-3

    def tearDown(self):
        reset_context()

    def _new_attention(self, backend: str, k_cache: torch.Tensor, v_cache: torch.Tensor) -> Attention:
        os.environ["NANOVLLM_ATTENTION_BACKEND"] = backend
        attn = Attention(self.num_heads, self.head_dim, self.scale, self.num_kv_heads).to(self.device)
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        return attn

    def test_prefill_outputs_are_identical(self):
        seqlens = [128, 96]
        total_tokens = sum(seqlens)

        q = torch.randn(total_tokens, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
        k = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v = torch.randn(total_tokens, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)

        cu_seqlens = torch.tensor([0, seqlens[0], total_tokens], dtype=torch.int32, device=self.device)
        slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device=self.device)

        block_size = 256
        num_blocks = 4
        k_cache = torch.zeros(num_blocks, block_size, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v_cache = torch.zeros_like(k_cache)

        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(seqlens),
            max_seqlen_k=max(seqlens),
            slot_mapping=slot_mapping,
            block_tables=None,
        )

        attn_flash_attn = self._new_attention("flash-attn", k_cache.clone(), v_cache.clone())
        attn_flashinfer = self._new_attention("flashinfer", k_cache.clone(), v_cache.clone())

        out_flash_attn = attn_flash_attn(q, k, v)
        out_flashinfer = attn_flashinfer(q, k, v)

        torch.testing.assert_close(out_flashinfer, out_flash_attn, rtol=self.rtol, atol=self.atol)

    def test_prefill_with_prefix_cache_outputs_are_identical(self):
        total_seqlens = [300, 385]
        cached_seqlens = [256, 320]
        query_seqlens = [total - cached for total, cached in zip(total_seqlens, cached_seqlens)]
        total_query_tokens = sum(query_seqlens)

        q = torch.randn(total_query_tokens, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)

        k_prefix_0 = torch.randn(cached_seqlens[0], self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v_prefix_0 = torch.randn_like(k_prefix_0)
        k_new_0 = torch.randn(query_seqlens[0], self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v_new_0 = torch.randn_like(k_new_0)

        k_prefix_1 = torch.randn(cached_seqlens[1], self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v_prefix_1 = torch.randn_like(k_prefix_1)
        k_new_1 = torch.randn(query_seqlens[1], self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v_new_1 = torch.randn_like(k_new_1)

        k = torch.cat([k_new_0, k_new_1], dim=0)
        v = torch.cat([v_new_0, v_new_1], dim=0)

        block_size = 256
        num_blocks = 4
        block_tables = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32, device=self.device)
        k_cache = torch.zeros(num_blocks, block_size, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v_cache = torch.zeros_like(k_cache)

        k_cache[0, :cached_seqlens[0]] = k_prefix_0
        v_cache[0, :cached_seqlens[0]] = v_prefix_0
        k_cache[1] = k_prefix_1[:block_size]
        v_cache[1] = v_prefix_1[:block_size]
        k_cache[2, :cached_seqlens[1] - block_size] = k_prefix_1[block_size:]
        v_cache[2, :cached_seqlens[1] - block_size] = v_prefix_1[block_size:]

        cu_seqlens_q = torch.tensor([0, query_seqlens[0], total_query_tokens], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, total_seqlens[0], sum(total_seqlens)], dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor(
            list(range(block_tables[0, 1].item() * block_size, block_tables[0, 1].item() * block_size + query_seqlens[0])) +
            list(range(block_tables[1, 1].item() * block_size + (cached_seqlens[1] - block_size), block_tables[1, 1].item() * block_size + (cached_seqlens[1] - block_size) + query_seqlens[1])),
            dtype=torch.int64,
            device=self.device,
        )

        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(query_seqlens),
            max_seqlen_k=max(total_seqlens),
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )

        attn_flash_attn = self._new_attention("flash-attn", k_cache.clone(), v_cache.clone())
        attn_flashinfer = self._new_attention("flashinfer", k_cache.clone(), v_cache.clone())

        out_flash_attn = attn_flash_attn(q, k, v)
        out_flashinfer = attn_flashinfer(q, k, v)

        torch.testing.assert_close(out_flashinfer, out_flash_attn, rtol=self.rtol, atol=self.atol)

    def test_decode_outputs_are_identical(self):
        # Two sequences: len=257 with block table [0, 3], len=384 with block table [1, 2]
        # flash-attn paged KV requires block_size to be divisible by 256.
        context_lens = torch.tensor([257, 384], dtype=torch.int32, device=self.device)
        block_tables = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32, device=self.device)
        slot_mapping = torch.tensor([768, 639], dtype=torch.int64, device=self.device)

        q = torch.randn(2, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
        k = torch.randn(2, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v = torch.randn(2, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)

        block_size = 256
        num_blocks = 4
        k_cache = torch.randn(num_blocks, block_size, self.num_kv_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v_cache = torch.randn_like(k_cache)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        attn_flash_attn = self._new_attention("flash-attn", k_cache.clone(), v_cache.clone())
        attn_flashinfer = self._new_attention("flashinfer", k_cache.clone(), v_cache.clone())

        out_flash_attn = attn_flash_attn(q, k, v)
        out_flashinfer = attn_flashinfer(q, k, v)

        if out_flash_attn.dim() == 4 and out_flash_attn.size(1) == 1:
            out_flash_attn = out_flash_attn.squeeze(1)
        torch.testing.assert_close(out_flashinfer, out_flash_attn, rtol=self.rtol, atol=self.atol)


if __name__ == "__main__":
    unittest.main()
