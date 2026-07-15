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

Confirmed working. The model correctly describes images sent via the OpenAI-compatible API and Open WebUI. This is the first TP3 deployment on RTX 6000 Pro with working multimodal input (that I am aware of anyway).

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
  -v /home/user/data/models:/models \
  -v /home/user/data/cache/minimax-m3:/root/.cache \
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
