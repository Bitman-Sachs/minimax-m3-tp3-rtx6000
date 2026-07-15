```bash
#!/bin/bash
# Launch MiniMax-M3 TP3 on 3× RTX PRO 6000 Blackwell
# Run inside the container after applying patch_dist_utils.py

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

echo "vLLM launch initiated. Monitor with:"
echo "  docker exec minimax-m3 tail -f /root/vllm.log"
```
