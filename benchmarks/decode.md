# Decode Throughput Benchmark

Measured with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) (30 seconds per cell, OpenAI stream usage).

## Aggregate Decode tok/s

| Context | C=1 | C=2 | C=3 |
|---------|-----|-----|-----|
| 0       | 71.6 | 114.3 | 139.5 |
| 8K      | 67.7 | 109.2 | 135.8 |
| 32K     | 60.3 | — | — |
| 64K     | 52.1 | 89.1 | 111.6 |
| 128K    | 41.8 | — | — |
| 195K    | 33.5 | — | — |

∅ = doesn't fit in 240K KV cache budget at that concurrency.

## Per-Request tok/s

| Context | C=1 | C=2 | C=3 |
|---------|-----|-----|-----|
| 0       | 71.6 | 57.2 | 46.5 |
| 8K      | 67.7 | 54.6 | 45.3 |
| 32K     | 60.3 | — | — |
| 64K     | 52.1 | 44.5 | 37.2 |
| 128K    | 41.8 | — | — |
| 195K    | 33.5 | — | — |

## Hardware During Benchmark

| Metric | Value |
|--------|-------|
| GPU avg util | 97-100% |
| VRAM usage | 98.4% |
| Avg power | 732W (3 GPUs combined) |
| Max power | 889W |
| Max temp | 95°C (CPU), 90°C (GPU) |

## Command

```bash
python3 llm_decode_bench.py \
  --port 8000 \
  --model minimax-m3 \
  --concurrency 1,2,3 \
  --contexts 0,8192,32768,65536,131072,200000 \
  --max-tokens 512 \
  --duration 30 \
  --kv-budget 240000 \
  --skip-prefill
```
