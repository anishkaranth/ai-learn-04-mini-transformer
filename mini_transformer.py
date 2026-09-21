"""Tiny character-level transformer (mini-GPT style) in NumPy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

@dataclass
class MiniTransformerConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    max_len: int = 64

class MiniTransformer:
    def __init__(self, cfg: MiniTransformerConfig, rng: np.random.Generator):
        self.cfg = cfg; self.rng = rng
        d, h = cfg.d_model, cfg.n_heads; assert d % h == 0
        self.tok_emb = rng.normal(0, 0.02, size=(cfg.vocab_size, d))
        self.pos_emb = rng.normal(0, 0.02, size=(cfg.max_len, d))
        self.layers = []
        for _ in range(cfg.n_layers):
            self.layers.append({'Wq': rng.normal(0,0.02,(d,d)), 'Wk': rng.normal(0,0.02,(d,d)), 'Wv': rng.normal(0,0.02,(d,d)), 'Wo': rng.normal(0,0.02,(d,d)), 'W1': rng.normal(0,0.02,(d,cfg.d_ff)), 'b1': np.zeros(cfg.d_ff), 'W2': rng.normal(0,0.02,(cfg.d_ff,d)), 'b2': np.zeros(d)})
        self.W_out = rng.normal(0, 0.02, size=(d, cfg.vocab_size)); self.b_out = np.zeros(cfg.vocab_size)
    def _causal_attn(self, x, layer):
        T, D = x.shape; H = self.cfg.n_heads; dh = D // H
        Q = (x @ layer['Wq']).reshape(T, H, dh).transpose(1,0,2)
        K = (x @ layer['Wk']).reshape(T, H, dh).transpose(1,0,2)
        V = (x @ layer['Wv']).reshape(T, H, dh).transpose(1,0,2)
        scores = (Q @ K.transpose(0,2,1)) / np.sqrt(dh)
        scores[:, np.triu(np.ones((T,T), dtype=bool), k=1)] = -1e9
        out = (softmax(scores, axis=-1) @ V).transpose(1,0,2).reshape(T, D)
        return out @ layer['Wo']
    def forward(self, idx):
        T = idx.shape[0]; x = self.tok_emb[idx] + self.pos_emb[:T]
        for layer in self.layers:
            x = layer_norm(x + self._causal_attn(x, layer))
            h = np.maximum(0.0, x @ layer['W1'] + layer['b1'])
            x = layer_norm(x + (h @ layer['W2'] + layer['b2']))
        return x @ self.W_out + self.b_out

def cross_entropy(logits, targets):
    probs = softmax(logits, axis=-1)
    return float(-np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-12)))

def numerical_grad_step(model, idx, targets, lr):
    logits = model.forward(idx); loss = cross_entropy(logits, targets)
    T = logits.shape[0]; probs = softmax(logits, axis=-1); dlogits = probs; dlogits[np.arange(T), targets] -= 1.0; dlogits /= T
    x = model.tok_emb[idx] + model.pos_emb[:T]
    model.W_out -= lr * (x.T @ dlogits); model.b_out -= lr * dlogits.sum(axis=0)
    dx = dlogits @ model.W_out.T
    for t in range(T):
        model.tok_emb[idx[t]] -= lr * dx[t]; model.pos_emb[t] -= lr * 0.1 * dx[t]
    return loss

def generate(model, start, n_new, temperature=1.0):
    idx = start.copy()
    for _ in range(n_new):
        ctx = idx[-model.cfg.max_len:]
        probs = softmax(model.forward(ctx)[-1] / max(temperature, 1e-6))
        idx = np.concatenate([idx, [model.rng.choice(len(probs), p=probs)]])
    return idx
