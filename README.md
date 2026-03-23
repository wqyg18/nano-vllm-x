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
# auto: prefer flashinfer when installed, otherwise flash-attn
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