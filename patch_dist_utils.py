```python
#!/usr/bin/env python3
"""
TP3 Patch for MiniMax-M3 Vision Tower on vLLM chthonic-consecration build.

Patches vllm/distributed/utils.py to replicate non-divisible components
(like the 16-head vision tower) across all TP ranks instead of crashing.

Usage:
    docker exec -i minimax-m3 python3 - < patch_dist_utils.py

Or pipe directly:
    cat patch_dist_utils.py | docker exec -i minimax-m3 python3 -
"""

import sys

def main():
    p = '/opt/venv/lib/python3.12/site-packages/vllm/distributed/utils.py'
    
    with open(p) as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    patched_divisibility = False
    patched_divide = False
    
    while i < len(lines):
        line = lines[i]
        
        # Patch 1: Replace ensure_divisibility with a no-op
        if 'def ensure_divisibility' in line and not patched_divisibility:
            new_lines.append('def ensure_divisibility(numerator, denominator):\n')
            new_lines.append('    pass  # patched for TP3\n')
            new_lines.append('\n')
            i += 1
            # Skip the original function body
            while i < len(lines):
                if lines[i].startswith('def ') or (lines[i].strip() == '' and i+1 < len(lines) and lines[i+1].startswith('def ')):
                    break
                i += 1
            patched_divisibility = True
            continue
        
        # Patch 2: Replace divide() to replicate when not divisible
        if 'def divide(' in line and not patched_divide:
            new_lines.append('def divide(numerator, denominator):\n')
            new_lines.append('    if numerator % denominator != 0:\n')
            new_lines.append('        return numerator\n')
            new_lines.append('    return numerator // denominator\n')
            new_lines.append('\n')
            i += 1
            # Skip the original function body
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
    
    if not patched_divisibility or not patched_divide:
        print("WARNING: One or both functions were not found. Check the file manually.")
        sys.exit(1)

if __name__ == '__main__':
    main()
```
