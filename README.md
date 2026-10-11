# Python-GPT

A decoder-only, GPT-style Transformer language model built from scratch in PyTorch and trained at character level on Tiny Shakespeare. The repo walks the full path from a bigram baseline to multi-head causal self-attention with residual blocks, LayerNorm, and dropout.

---

## Highlights

- Decoder-only Transformer (GPT architecture), ~10.79M parameters
- Multi-head causal self-attention (6 heads) with lower-triangular masking
- Pre-LayerNorm residual blocks with a 4x feed-forward MLP
- Learned token and positional embeddings
- Autoregressive character-level text generation
- Train / validation split with periodic loss estimation
- Trained on a Tesla T4 GPU for 5000 iterations
- Checkpoint (weights, optimizer state, vocab, config) stored with Git LFS

---

## Model Architecture

| Component | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Attention | Multi-head causal self-attention |
| Layers (`n_layer`) | 6 |
| Attention heads (`n_head`) | 6 |
| Embedding dim (`n_embd`) | 384 |
| Head size | 384 / 6 = 64 |
| Context length (`block_size`) | 256 |
| Vocabulary size | 65 (character-level) |
| Feed-forward | 384 -> 1536 -> 384, ReLU |
| Normalization | Pre-LayerNorm + final LayerNorm |
| Dropout | 0.2 |
| Weight init | Normal (std 0.02) |
| Total parameters | ~10.79M |

```
Input characters
      |
Token Embedding + Positional Embedding
      |
+-----v----------------------------+
|  Transformer Block  (x 6)        |
|   LayerNorm                      |
|   Multi-Head Causal Self-Attn    |
|   + Residual                     |
|   LayerNorm                      |
|   Feed-Forward (MLP)             |
|   + Residual                     |
+-----+----------------------------+
      |
Final LayerNorm
      |
Linear LM head -> logits over 65 characters
```

---

## Training Setup

| Setting | Value |
|---|---|
| Dataset | Tiny Shakespeare (1,115,394 characters) |
| Tokenizer | Character-level (65 unique characters) |
| Train / Val split | 90% / 10% |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 3e-4 (constant) |
| Iterations | 5000 |
| Eval interval | every 500 iterations (200 batches averaged) |
| Loss function | Cross-entropy (next-character prediction) |
| Seed | 1337 |
| Framework | PyTorch |

### Hardware

| | |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Training length | 5000 iterations |

---

## Results

| Metric | Value |
|---|---|
| Final train loss | 1.13 |
| Final validation loss | 1.48 |

---

## Installation

```bash
git lfs install
git clone https://github.com/murshadaziz/Python-GPT.git
cd Python-GPT
pip install torch
```

Git LFS is required to download the trained checkpoint (`gpt_checkpoint.pth`).

---

## Usage

**Download the dataset** (the notebook does this automatically):

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -P data/
```

**Generate text from the trained checkpoint**

```bash
python GPT.py
```

This loads `gpt_checkpoint.pth`, prints the parameter count, and generates 500 characters.

**Retrain from scratch:** uncomment the training loop and the `torch.save` block in `GPT.py`, and remove the checkpoint-loading section.

---

## Project Structure

```
Python-GPT/
├── GPT.py              # full GPT model, training loop, checkpoint loading, generation
├── GPT.ipynb           # step-by-step notebook: data, bigram, self-attention, full model
├── bigram.py           # bigram language model baseline
├── gpt_checkpoint.pth  # trained model (Git LFS)
├── data/               # dataset (input.txt)
└── README.md
```

---

## Future Work

- Subword tokenization (BPE)
- Mixed precision (fp16/bf16) and gradient accumulation
- KV-cache for faster generation
- Temperature / top-k sampling
- Larger dataset, context length, and model size

---

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Andrej Karpathy, *Let's build GPT* and *nanoGPT*
