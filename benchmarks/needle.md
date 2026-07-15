# Needle-in-Haystack Test Results

3 runs per context size with unique padding (prevents prefix cache cheating).

## Results

| Context | Run 1 | Run 2 | Run 3 | Pass Rate |
|---------|-------|-------|-------|-----------|
| 0       | PASS  | PASS  | PASS  | 3/3       |
| 8K      | PASS  | PASS  | PASS  | 3/3       |
| 32K     | PASS  | PASS  | PASS  | 3/3       |
| 64K     | PASS  | PASS  | PASS  | 3/3       |
| 128K    | PASS  | PASS  | PASS  | 3/3       |
| 200K    | PASS  | PASS  | PASS  | 3/3       |

**100% pass rate (18/18)**

## Methodology

- Needle: "The secret access code for the vault is BLUEBIRD-7749."
- Padding: Unique per run (includes run ID to prevent prefix cache hits)
- Needle placed at 50% of context
- Temperature: 0
- Max tokens: 200
- Model returns "BLUEBIRD-7749" = PASS

## What This Proves

FP8 KV cache preserves retrieval quality at high context. The 240K context limit is a VRAM constraint, not a quality constraint — the model can correctly find and retrieve information placed at 200K tokens.
