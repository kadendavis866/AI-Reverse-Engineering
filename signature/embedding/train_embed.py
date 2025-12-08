"""
Train a skip-gram (negative sampling) embedding model for instructions using PyTorch.

This version streams (center, context) pairs from the token sequence to avoid
materializing all pairs in memory. It uses an IterableDataset and modest defaults
to keep RAM/VRAM use low.
"""

import argparse
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


class SkipGramIterable(IterableDataset):
    def __init__(self, tokens: List[int], window_size: int):
        self.tokens = tokens
        self.window = window_size

    def __iter__(self):
        tokens = self.tokens
        w = self.window
        n = len(tokens)
        for i in range(n):
            start = max(0, i - w)
            end = min(n, i + w + 1)
            for j in range(start, end):
                if j == i:
                    continue
                yield tokens[i], tokens[j]


class SGNSModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.input_emb.weight, -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.zeros_(self.output_emb.weight)

    def forward(self, center, pos, neg):
        v = self.input_emb(center)          # (B, D)
        u_pos = self.output_emb(pos)        # (B, D)
        u_neg = self.output_emb(neg)        # (B, K, D)

        pos_score = torch.sum(v * u_pos, dim=1)          # (B,)
        pos_loss = F.logsigmoid(pos_score)               # (B,)

        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = F.logsigmoid(-neg_score).sum(1)               # (B,)

        return -(pos_loss + neg_loss).mean()


def load_tokens(path: str) -> List[int]:
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    return [int(tok) for tok in text.split()]


def build_vocab(tokens: List[int], min_count: int):
    counter = Counter(tokens)
    items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_count]
    items.sort(key=lambda x: -x[1])
    id2word = [tok for tok, _ in items]
    word2id = {tok: i for i, tok in enumerate(id2word)}
    freqs = torch.tensor([cnt for _, cnt in items], dtype=torch.float)
    return word2id, id2word, freqs


def subsample_tokens(token_ids: List[int], freqs: torch.Tensor, threshold: float):
    """Subsample frequent tokens; operates on contiguous token ids."""
    if threshold <= 0:
        return token_ids
    total = freqs.sum().item()
    probs = freqs / total
    probs = torch.clamp(1.0 - torch.sqrt(threshold / probs), min=0.0, max=1.0)
    kept = []
    for t in token_ids:
        if torch.rand(1).item() > probs[t].item():
            kept.append(t)
    return kept


def make_neg_sampler(freqs: torch.Tensor):
    weights = freqs.pow(0.75)
    dist = weights / weights.sum()

    def sample(batch_size, num_neg):
        return torch.multinomial(dist, batch_size * num_neg, replacement=True)

    return sample


def get_output_paths(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cnt = 1
    output_path = output_dir / f"embed_{cnt}.emb"
    while output_path.exists():
        cnt += 1
        output_path = output_dir / f"embed_{cnt}.emb"
    model_folder = output_dir / f"model_{cnt}"
    model_folder.mkdir(parents=True, exist_ok=True)
    return output_path, model_folder


def train(args):
    tokens = load_tokens(args.input_path)
    if not tokens:
        raise ValueError("No tokens loaded from input_path.")

    word2id, id2word, freqs = build_vocab(tokens, args.min_count)
    tokens = [word2id[t] for t in tokens if t in word2id]
    tokens = subsample_tokens(tokens, freqs, args.subsample)
    print(int(len(tokens)*2*args.window_size/args.batch_size), "approx total steps per epoch")
    dataset = SkipGramIterable(tokens, args.window_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGNSModel(len(word2id), args.embed_dim).to(device)
    neg_sampler = make_neg_sampler(freqs)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10,
                                                                min_lr=1e-6)


    step = 0
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}/{args.num_epochs}")
        total_loss = 0.0
        for centers, pos in dataloader:
            batch_size = centers.size(0)
            neg = neg_sampler(batch_size, args.num_neg_samples).view(batch_size, args.num_neg_samples)
            centers = centers.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            loss = model(centers, pos, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs} Step {step} Loss {loss.item():.4f}")
        scheduler.step(total_loss)
    output_dir = Path(args.output_dir)
    output_path, model_folder = get_output_paths(output_dir)

    ckpt_path = model_folder / "model.pt"
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(word2id), "embed_dim": args.embed_dim}, ckpt_path)

    insn_embed = {
        "vocab_size": len(word2id),
        "embedding_size": args.embed_dim,
        "word2id": word2id,
        "id2word": id2word,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "num_neg_samples": args.num_neg_samples,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "min_count": args.min_count,
        "subsample": args.subsample,
        "embeddings": model.input_emb.weight.detach().cpu().numpy(),
    }
    with open(output_path, "wb") as f:
        pickle.dump(insn_embed, f)
    print(f"Saved embedding to {output_path} and checkpoint to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="embed_input", help="Input file produced by prep_embed.py (space-separated ints).")
    parser.add_argument("-o", "--output_dir", default="embed_output", help="Directory to save embeddings/checkpoints.")
    parser.add_argument("-e", "--embed_dim", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("-ne", "--num_epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("-nn", "--num_neg_samples", type=int, default=2, help="Negative samples per positive.")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("-ws", "--window_size", type=int, default=2, help="Context window size.")
    parser.add_argument("-mc", "--min_count", type=int, default=1, help="Discard tokens with freq < min_count.")
    parser.add_argument("-s", "--subsample", type=float, default=0.05, help="Subsampling threshold (<=0 to disable).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
