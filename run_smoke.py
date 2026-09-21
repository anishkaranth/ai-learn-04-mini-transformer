#!/usr/bin/env python3
import json, time
from pathlib import Path
import numpy as np
from mini_transformer import MiniTransformer, MiniTransformerConfig, numerical_grad_step
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
def main():
    rng = np.random.default_rng(42)
    text = (ROOT / 'data' / 'tiny.txt').read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    cfg = MiniTransformerConfig(vocab_size=len(stoi), d_model=48, n_heads=4, n_layers=2, d_ff=96, max_len=48)
    model = MiniTransformer(cfg, rng)
    losses = []
    t0 = time.time()
    for step in range(200):
        start = int(rng.integers(0, max(1, len(data) - 33)))
        idx = data[start:start + 32]
        tgt = data[start + 1:start + 33]
        losses.append(numerical_grad_step(model, idx, tgt, lr=0.15))
    RESULTS.mkdir(exist_ok=True)
    shot = {'snapshot': 'smoke_run', 'seed': 42, 'final_loss': float(losses[-1]), 'mean_last_50': float(np.mean(losses[-50:])), 'vocab_size': len(stoi), 'runtime_s': round(time.time() - t0, 3)}
    (RESULTS / 'JSON.shot').write_text(json.dumps(shot, indent=2) + '\n')
    print(json.dumps(shot, indent=2))
if __name__ == '__main__':
    main()
