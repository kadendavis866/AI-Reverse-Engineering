"""
Predict function signatures (e.g., num_args) using the trained GRU classifier.

Inputs:
- A JSONL with per-function entries containing inst_bytes (as lists of bytes).
- A trained signature model checkpoint from sig_train.py.
- An embedding pickle from signature/embedding/train_embed.py.

Output:
- JSONL with per-function predictions: addr_start (if provided), label_name (if provided), predicted_label.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn

from signature.embedding import insn_int


def load_embeddings(embed_path: str):
    data = pickle.load(open(embed_path, "rb"))
    emb_matrix = torch.tensor(data["embeddings"], dtype=torch.float)
    word2id = {int(k): int(v) for k, v in data["word2id"].items()}
    pad_row = torch.zeros(1, emb_matrix.shape[1], dtype=torch.float)
    emb_matrix = torch.cat([emb_matrix, pad_row], dim=0)
    pad_idx = emb_matrix.shape[0] - 1
    return emb_matrix, word2id, pad_idx


class GRUClassifier(nn.Module):
    def __init__(self, emb_matrix: torch.Tensor, hidden_size: int, num_layers: int, num_classes: int, pad_idx: int,
                 dropout: float):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=emb_matrix.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # (B, L, D)
        out, _ = self.gru(emb)  # (B, L, H)
        last = out[:, -1, :]  # use last timestep
        logits = self.fc(last)
        return logits


def insns_to_ids(inst_bytes: List[bytes], word2id, pad_idx: int, max_length: int) -> List[int]:
    tokens = []
    for insn in inst_bytes:
        token = insn_int.insn2int_inverse(insn)
        tokens.append(word2id.get(token, pad_idx))
        if len(tokens) >= max_length:
            break
    if len(tokens) < max_length:
        tokens.extend([pad_idx] * (max_length - len(tokens)))
    return tokens


def load_jsonl_funcs(path: str) -> List[Dict]:
    funcs = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, dict) and "inst_bytes" in obj:
                funcs.append(obj)
    return funcs


def predict(funcs: List[Dict], model, word2id, pad_idx, max_length, device):
    results = []
    model.eval()
    with torch.no_grad():
        for func in funcs:
            tokens = insns_to_ids(func["inst_bytes"], word2id, pad_idx, max_length)
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            results.append({
                "addr_start": hex(int(func.get("addr_start"))),
                "label_name": "num_args",
                "predicted_label": pred,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict function signatures from embeddings and a trained GRU.")
    parser.add_argument("--input", required=True, help="JSONL with inst_bytes per function.")
    parser.add_argument("--embed-path", required=True, help="Embedding pickle from train_embed.py.")
    parser.add_argument("--model-path", required=True, help="GRU checkpoint from sig_train.py.")
    parser.add_argument("--max-length", type=int, default=200, help="Max instructions per function.")
    parser.add_argument("--output", default="sig_preds.jsonl", help="Output JSONL file.")
    args = parser.parse_args()

    emb_matrix, word2id, pad_idx = load_embeddings(args.embed_path)
    ckpt = torch.load(args.model_path, map_location="cpu")
    hidden_size = ckpt.get("hidden_size", 128)
    num_layers = ckpt.get("num_layers", 2)
    num_classes = ckpt.get("num_classes", ckpt["state_dict"]["fc.weight"].shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(emb_matrix, hidden_size, num_layers, num_classes, pad_idx, dropout=0.0)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    funcs = load_jsonl_funcs(args.input)
    if not funcs:
        raise ValueError("No functions loaded from input.")

    preds = predict(funcs, model, word2id, pad_idx, args.max_length, device)
    out_path = Path(args.output)
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        for row in preds:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
