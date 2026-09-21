# Smoke results — ai-learn-04-mini-transformer

**Seed:** `42` · char-level mini-GPT (NumPy)

## Headline metrics

| Metric | Value |
|--------|------:|
| Vocab size | 25 |
| Steps | 400 |
| Initial loss | 3.2250 |
| Final loss | 2.7890 |
| Mean last-50 loss | 2.3850 |
| Best loss | 1.9381 |
| Runtime (s) | 0.452 |

## Plots

- [`loss_curve.png`](loss_curve.png) / [`loss_curve.svg`](loss_curve.svg)

## Takeaway

A tiny causal transformer with a lightweight output-projection + embedding update drives CE down on a tiny corpus.
