# AI Learn 04 — Mini Transformer (char-level)

A tiny **decoder-only transformer** (mini-GPT style) in **NumPy**: causal self-attention, train loop on an in-repo text file, loss curve, and temperature samples.

## Learning goals

- Causal mask in self-attention
- Token + positional embeddings
- Next-character prediction CE loss
- How temperature changes generated text

## Layout

```
mini_transformer.py
data/tiny.txt
run_smoke.py
notebooks/mini_transformer.ipynb
results/
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

> Educational train step updates the output projection + embeddings (lightweight). Enough to drive loss down on a tiny corpus without PyTorch.
