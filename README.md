fork from Nano-vLLM

quick-start:

```
uv sync
```


run some scripts:
 
```
uv run python -m path.file
uv run python -m file.py
```

attention backend:

```
# current auto: prefer flashinfer when installed, otherwise flash-attn
NANOVLLM_ATTENTION_BACKEND=auto uv run python example.py

# force flash-attn
NANOVLLM_ATTENTION_BACKEND=flash-attn uv run python example.py

# force flashinfer
NANOVLLM_ATTENTION_BACKEND=flashinfer uv run python example.py
```

test:
```
uv run python -m unittest discover -s tests
```

todo:

- extend attention throughput tests to sweep batch size, context len, and prefill len, then map the crossover points between flash-attn and flashinfer
- replace the current `auto` backend policy with a heuristic selector based on workload shape instead of simply preferring flashinfer when installed
