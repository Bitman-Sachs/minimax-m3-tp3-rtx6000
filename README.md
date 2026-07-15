# MiniMax-M3 TP3 on 3× RTX PRO 6000 Blackwell

Running MiniMax-M3 (428B MoE, 23B active) across 3× RTX PRO 6000 Blackwell (96 GB) at tensor parallelism 3 (TP=3) with 240K context, FP8 KV cache, and **working multimodal vision input**.

## The Problem

MiniMax-M3 has 64 LLM attention heads and 16 vision tower attention heads. Neither divides evenly by 3:
- 64 ÷ 3 = 21.33 (handled by Luke Alonso's virtual sharding — pads to 96)
- **16 ÷ 3 = 5.33 (crashes vLLM with `AssertionError: 16 is not divisible by 3`)**

The current pinned Docker image (`f1190eab`) includes the vision tower code. When vLLM tries to shard 16 vision heads across 3 GPUs, it crashes before any weights load.

## The Solution

A 4-line patch to vLLM's `dist_utils.py` that makes the `divide()` function **replicate** non-divisible components instead of crashing. The vision tower (~1-2 GB) is replicated across all 3 GPUs instead of sharded. The main LLM is still properly sharded by Luke's virtual sharding.

This is different from the workaround of using an older Docker image (`2cdc3d9`) that simply doesn't have vision tower code. That approach works for text-only but **cannot process images**. This patch works with any image version and enables full multimodal input.

## Key Results

### Decode Throughput (llm-inference-bench, 30s per cell)

| Context | C=1 tok/s | C=2 tok/s | C=3 tok/s |
|---------|-----------|-----------|-----------|
| 0       | 71.6      | 114.3     | 139.5     |
| 8K      | 67.7      | 109.2     | 135.8     |
| 32K     | 60.3      | —         | —         |
| 64K     | 52.1      | 89.1      | 111.6     |
| 128K    | 41.8      | —         | —         |
| 195K    | 33.5      | —         | —         |

∅ = doesn't fit in 240K KV cache budget at that concurrency.

### Needle-in-Haystack (3 runs per context, unique padding)

| Context | Pass Rate |
|---------|-----------|
| 0       | 3/3       |
| 8K      | 3/3       |
| 32K     | 3/3       |
| 64K     | 3/3       |
| 128K    | 3/3       |
| 200K    | 3/3       |

**100% pass rate (18/18)** — FP8 KV cache preserves retrieval quality at high context.

### Vision Input

Confirmed working. The model correctly describes images sent via the OpenAI-compatible API and Open WebUI. This is the first TP3 deployment on RTX 6000 Pro with working multimodal input.

### Prefill (~100K context, from vLLM engine logs)

~3,200 tok/s average (comparable to community benchmarks at similar context).

## Hardware

- AMD Threadripper PRO (WRX90 chipset), 256 GB RAM
- 3× RTX PRO 6000 Blackwell Max-Q (96 GB each)
- PCIe Gen5 (no NVLink)

## Quick Start

### Step 1 — Create container

```bash
docker run -d --name minimax-m3 \
  --gpus '"device=0,1,2"' \
  --ipc=host --shm-size=64g --cap-add SYS_NICE \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/your/models:/models \
  -v /path/to/your/cache:/root/.cache \
  -p 8000:8000 \
  -e HOME=/root \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_SHM_DISABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_ALGO=Ring \
  -e NCCL_PROTO=Simple \
  -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_MAX_NCHANNELS=1 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=8 \
  -e VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_B12X_FP8_GEMM=0 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_MINIMAX_M3_MSA=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0 \
  --entrypoint /bin/bash \
  voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132 \
  -c "sleep infinity"
```

### Step 2 — Apply TP3 patch

```bash
docker exec -i minimax-m3 python3 - << 'PYEOF'
p = '/opt/venv/lib/python3.12/site-packages/vllm/distributed/utils.py'
with open(p) as f:
    lines = f.readlines()
new_lines = []
i = 0
patched_divisibility = False
patched_divide = False
while i < len(lines):
    line = lines[i]
    if 'def ensure_divisibility' in line and not patched_divisibility:
        new_lines.append('def ensure_divisibility(numerator, denominator):\n')
        new_lines.append('    pass  # patched for TP3\n')
        new_lines.append('\n')
        i += 1
        while i < len(lines):
            if lines[i].startswith('def ') or (lines[i].strip() == '' and i+1 < len(lines) and lines[i+1].startswith('def ')):
                break
            i += 1
        patched_divisibility = True
        continue
    if 'def divide(' in line and not patched_divide:
        new_lines.append('def divide(numerator, denominator):\n')
        new_lines.append('    if numerator % denominator != 0:\n')
        new_lines.append('        return numerator\n')
        new_lines.append('    return numerator // denominator\n')
        new_lines.append('\n')
        i += 1
        while i < len(lines):
            if lines[i].startswith('def ') or (lines[i].startswith('class ')) or (lines[i].strip() == '' and i+1 < len(lines) and (lines[i+1].startswith('def ') or lines[i+1].startswith('class '))):
                break
            i += 1
        patched_divide = True
        continue
    new_lines.append(line)
    i += 1
with open(p, 'w') as f:
    f.writelines(new_lines)
print(f"PATCHED: ensure_divisibility={patched_divisibility}, divide={patched_divide}")
PYEOF
```

### Step 3 — Launch vLLM

```bash
docker exec -d minimax-m3 bash -c 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS && vllm serve \
  --model /models/lukealonso-MiniMax-M3-MXFP8-NVFP4 \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 3 \
  --quantization modelopt_mixed \
  --kv-cache-dtype fp8 \
  --attention-backend B12X_ATTN \
  --moe-backend b12x \
  --gpu-memory-utilization 0.97 \
  --max-model-len 240000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --block-size 128 \
  --load-format fastsafetensors \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --override-generation-config "{\"temperature\":1,\"top_p\":0.95,\"top_k\":40}" \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  > /root/vllm.log 2>&1'
```

### Step 4 — Monitor startup (~5-7 minutes)

```bash
docker exec minimax-m3 tail -f /root/vllm.log
```

Wait for `Application startup complete`.

### Step 5 — Verify

```bash
curl -s --max-time 30 -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m3","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}' \
  -w "\nHTTP:%{http_code} TIME:%{time_total}s"
```

## Configuration Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Docker image | `voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132` | Current pinned image with vision tower + b12x kernels |
| Model | `lukealonso/MiniMax-M3-MXFP8-NVFP4` | MXFP8/NVFP4 mixed quant (~243 GB) |
| TP size | 3 | 3 available GPUs |
| GPU memory util | 0.97 | Maximizes KV cache (10.5 GiB per GPU) |
| Max context | 240,000 | Calculated from KV cache capacity |
| KV cache dtype | fp8 | Halves KV memory vs BF16, doubles context |
| Attention backend | B12X_ATTN | CuTe-DSL kernels for SM120 |
| Reasoning parser | minimax_m3 | Handles thinking tokens |

## VRAM Budget (per GPU)

| Component | Memory |
|-----------|--------|
| Total VRAM | 97.8 GiB |
| Model weights | 80.1 GiB |
| CUDA graphs | 0.2 GiB |
| Vision tower (replicated) | ~1-2 GiB |
| **Available KV cache** | **~10.5 GiB** |
| **KV needed for 240K** | **~9.6 GiB** |
| **Safety margin** | **~0.9 GiB** |

## Stability

Running stably for ~1 week through agentic workflows and single-user prompting. Experienced 1-2 silent API server crashes in that period — likely a bug in the b12x vLLM fork, not related to the TP3 patch. A watchdog script (`watchdog.sh`) is included that checks server health every 30 seconds and auto-restarts if needed. This is probably overkill given the rarity of the crashes, but included for completeness.

## How the Patch Works

vLLM's `dist_utils.py` has a `divide()` function that shards components across GPUs. When a component's size isn't divisible by the TP size, it crashes:

```python
# Original (crashes):
def divide(numerator, denominator):
    assert numerator % denominator == 0  # 16 % 3 != 0 → CRASH
    return numerator // denominator

# Patched (works):
def divide(numerator, denominator):
    if numerator % denominator != 0:
        return numerator  # Replicate on all GPUs instead of sharding
    return numerator // denominator
```

This makes the vision tower (16 heads) replicate across all 3 GPUs instead of trying to shard 16 ÷ 3. The vision tower is tiny (~1-2 GB) vs the 428B main model (~80 GB per GPU), so replication costs negligible VRAM.

The main LLM (64 heads) is handled by Luke Alonso's virtual sharding (pads 64→96, then 96 ÷ 3 = 32 per GPU), which is already in the chthonic build.

## Credits

- **Luke Alonso** — chthonic vLLM fork with TP3 virtual sharding for the LLM heads, b12x kernels, and the NVFP4 quantization pipeline
- **Fruity** (RTX 6000 Pro Discord) — first to demonstrate TP3 on 3× RTX 6000 Pro using the older `2cdc3d9` image (text-only, no vision)
- **voipmonitor** — Docker image builds of the chthonic fork

## License

MIT
