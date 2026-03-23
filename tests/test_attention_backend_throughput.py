import os
import time
import unittest

import torch

from nanovllm.layers.attention import Attention, flashinfer
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for attention backend throughput tests")
@unittest.skipUnless(flashinfer is not None, "flashinfer is required for attention backend throughput tests")
class TestAttentionBackendThroughput(unittest.TestCase):
    warmup_iters = 10
    measure_iters = 50

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.num_heads = 8
        self.num_kv_heads = 8
        self.head_dim = 128
        self.scale = self.head_dim ** -0.5

    def tearDown(self):
        reset_context()

    def _new_attention(self, backend: str, k_cache: torch.Tensor, v_cache: torch.Tensor) -> Attention:
        os.environ["NANOVLLM_ATTENTION_BACKEND"] = backend
        attn = Attention(self.num_heads, self.head_dim, self.scale, self.num_kv_heads).to(self.device)
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        return attn

    def _benchmark(self, fn, total_tokens: int) -> tuple[float, float]:
        for _ in range(self.warmup_iters):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(self.measure_iters):
            fn()
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

        avg_latency_ms = elapsed_s * 1000 / self.measure_iters
        throughput_tok_s = total_tokens * self.measure_iters / elapsed_s
        return avg_latency_ms, throughput_tok_s

    def test_prefill_throughput(self):
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

        for backend in ("flash-attn", "flashinfer"):
            attn = self._new_attention(backend, k_cache.clone(), v_cache.clone())
            latency_ms, throughput_tok_s = self._benchmark(lambda: attn(q, k, v), total_tokens)
            print(
                f"\nprefill backend={backend} "
                f"latency_ms={latency_ms:.3f} "
                f"throughput_tok_s={throughput_tok_s:.2f}"
            )

    def test_decode_throughput(self):
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

        for backend in ("flash-attn", "flashinfer"):
            attn = self._new_attention(backend, k_cache.clone(), v_cache.clone())
            latency_ms, throughput_tok_s = self._benchmark(lambda: attn(q, k, v), q.size(0))
            print(
                f"\ndecode backend={backend} "
                f"latency_ms={latency_ms:.3f} "
                f"throughput_tok_s={throughput_tok_s:.2f}"
            )


if __name__ == "__main__":
    unittest.main()
