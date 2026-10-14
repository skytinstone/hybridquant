| model | mixer | L | L_a | L_g | hidden | H_kv | d | H | d_k | d_v | |KV|/token | |S| (M) | |S| bf16 MB | T* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | GDN | 24 | 6 | 18 | 1024 | 2 | 256 | 16 | 128 | 128 | 6,144 | 4.72 | 9.4 | 768 |
| Qwen3.5-2B | GDN | 24 | 6 | 18 | 2048 | 2 | 256 | 16 | 128 | 128 | 6,144 | 4.72 | 9.4 | 768 |
| Qwen3.5-4B | GDN | 32 | 8 | 24 | 2560 | 4 | 256 | 32 | 128 | 128 | 16,384 | 12.58 | 25.2 | 768 |
| Qwen3.5-9B | GDN | 32 | 8 | 24 | 4096 | 4 | 256 | 32 | 128 | 128 | 16,384 | 12.58 | 25.2 | 768 |
| Falcon-H1-1.5B-Instruct | Mamba-2 | 24 | 24 | 24 | 2048 | 2 | 128 | 48 | 64 | 256 | 12,288 | 18.87 | 37.7 | 1,536 |

|KV|/token = 2 L_a H_kv d (elements, K and V); |S| = L_g H d_k d_v (elements); T* = |S| / (|KV|/token) is the context length at which the two memories are equal. Falcon-H1 has attention and Mamba-2 in every layer (L_a = L_g = L); its state is [H, d_head, d_state] with d_k = d_head and d_v = d_state.
