```bash
#!/bin/bash
# Watchdog for MiniMax-M3 TP3 — checks server health every 30 seconds
# Auto-restarts vLLM if the API server dies (leaving zombie workers)
#
# Usage:
#   chmod +x watchdog.sh
#   nohup ./watchdog.sh > watchdog.log 2>&1 &

while true; do
  sleep 30
  
  if ! curl -s --max-time 5 -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"minimax-m3","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
    > /dev/null 2>&1; then
    
    echo "[$(date)] Server down — restarting vLLM..."
    
    # Kill zombie workers
    docker exec minimax-m3 pkill -9 -f vllm 2>/dev/null
    sleep 5
    
    # Restart vLLM
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
    
    echo "[$(date)] Restart initiated — waiting 5 min for startup..."
    sleep 300
  fi
done
```
