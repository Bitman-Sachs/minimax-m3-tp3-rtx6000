# Prefill Throughput

Measured from vLLM engine logs (`Avg prompt throughput`) during a ~100K token request.

## Results

| Context | Prefill tok/s (engine avg) |
|---------|---------------------------|
| ~100K   | ~3,200                    |

Engine log samples during the request:

```
Avg prompt throughput: 4096.0 tokens/s  (early in prefill)
Avg prompt throughput: 3276.8 tokens/s  (mid prefill)
Avg prompt throughput: 3686.3 tokens/s  (mid prefill)
Avg prompt throughput: 2867.2 tokens/s  (late prefill)
Avg prompt throughput: 2542.1 tokens/s  (near end)
```

Average across the full prefill: ~3,200 tok/s.

## Context

- Prefix cache hit rate: 93% (repetitive padding benefits from caching)
- Wall time for 74,435 prompt tokens: 20.6s
- FP8 KV cache, 0.97 GPU memory utilization

## Note

These numbers include prefix cache effects. Cold prefill (no cache hits) would be lower. The vLLM engine log reports the actual throughput including cache hits, which is what users experience in practice when system prompts are reused.
