"""
PyTorch training script for function signature prediction using instruction embeddings.

Inputs:
- data_folder: directory containing BinaryFileDef pickles (recursively).
- embed_path: pickle produced by signature/embedding/train_embed.py (contains embeddings, word2id).

The script builds token id sequences from inst_bytes via insn_int.insn2int_inverse, pads/truncates
to max_length, and trains a GRU classifier to predict num_args (default) or a custom label field.
"""

import argparse
import csv
import glob
import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from signature.embedding import insn_int  # adjust if package layout changes
from binary_file_format import BinaryFileDef


def load_embeddings(embed_path: str):
    data = pickle.load(open(embed_path, "rb"))
    emb_matrix = torch.tensor(data["embeddings"], dtype=torch.float)
    word2id = {int(k): int(v) for k, v in data["word2id"].items()}
    pad_row = torch.zeros(1, emb_matrix.shape[1], dtype=torch.float)
    emb_matrix = torch.cat([emb_matrix, pad_row], dim=0)
    pad_idx = emb_matrix.shape[0] - 1
    return emb_matrix, word2id, pad_idx


def insns_to_ids(func: BinaryFileDef.Function, word2id, pad_idx, max_length: int) -> List[int]:
    tokens = []
    for insn in func.inst_bytes:
        token = insn_int.insn2int_inverse(insn)
        tokens.append(word2id.get(token, pad_idx))
        if len(tokens) >= max_length:
            break
    if len(tokens) < max_length:
        tokens.extend([pad_idx] * (max_length - len(tokens)))
    return tokens


class FuncDataset(Dataset):
    def __init__(self, files: List[str], word2id, pad_idx: int, max_length: int, num_classes: int, label_field: str):
        self.samples: List[Tuple[List[int], int]] = []
        for path in files:
            binary: BinaryFileDef = pickle.load(open(path, "rb"))
            for func in binary.functions:
                tokens = insns_to_ids(func, word2id, pad_idx, max_length)
                if label_field == "num_args":
                    label = func.num_args
                else:
                    label = getattr(func, label_field, None)
                    if label is None or isinstance(label, (list, tuple, dict)):
                        continue
                label = min(label, num_classes - 1)
                self.samples.append((tokens, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class GRUClassifier(nn.Module):
    def __init__(self, emb_matrix: torch.Tensor, hidden_size: int, num_layers: int, num_classes: int, pad_idx: int, dropout: float):
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
        out, _ = self.gru(emb)   # (B, L, H)
        last = out[:, -1, :]     # use last timestep
        logits = self.fc(last)
        return logits


def train_model(args):
    emb_matrix, word2id, pad_idx = load_embeddings(args.embed_path)
    pickle_files = sorted(glob.glob(str(Path(args.data_folder) / "x64" / "*O0*.pkl")))
    if not pickle_files:
        raise FileNotFoundError(f"No pickles found under {args.data_folder}")

    dataset = FuncDataset(pickle_files, word2id, pad_idx, args.max_length, args.num_classes, args.label_tag)
    if len(dataset) == 0:
        raise ValueError("No samples built from pickles.")

    val_len = max(1, int(0.1 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(emb_matrix, args.hidden_size, args.num_layers, args.num_classes, pad_idx, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "sig_model.pt"
    csv_path = save_dir / "metrics.csv"

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_acc", "val_acc", "val_precision", "val_recall", "val_f1"])

    for epoch in range(args.epoch_num):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for tokens, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(tokens)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.numel()
        avg_train = total_loss / max(1, len(train_loader))
        train_acc = total_correct / max(1, total_examples)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens = tokens.to(device)
                labels = labels.to(device)
                logits = model(tokens)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
                tp += ((preds == labels) & (labels != 0)).sum().item()
                fp += ((preds != labels) & (preds != 0)).sum().item()
                fn += ((preds != labels) & (labels != 0)).sum().item()
        avg_val = val_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * precision * recall / max(1e-8, (precision + recall))

        print(f"Epoch {epoch+1}/{args.epoch_num} train_loss={avg_train:.4f} train_acc={train_acc:.4f} val_loss={avg_val:.4f} val_acc={val_acc:.4f} val_prec={precision:.4f} val_rec={recall:.4f} val_f1={f1:.4f}")
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, f"{train_acc:.4f}", f"{val_acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "state_dict": model.state_dict(),
                "pad_idx": pad_idx,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_classes": args.num_classes,
                "embed_dim": emb_matrix.shape[1],
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", default="../../data/pickles", help="Directory containing BinaryFileDef pickles.")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save logs/models.")
    parser.add_argument("-e", "--embed_path", default="../embedding/embed_output/embed_1.emb", help="Path to embedding pickle from train_embed.py.")
    parser.add_argument("-lt", "--label_tag", default="num_args", help="Label field to use (default num_args).")
    parser.add_argument("-ed", "--embedding_dim", type=int, default=128, help="Embedding dimension (informational).")
    parser.add_argument("-ml", "--max_length", type=int, default=200, help="Maximum tokens per function.")
    parser.add_argument("-nc", "--num_classes", type=int, default=16, help="Number of classes.")
    parser.add_argument("-en", "--epoch_num", type=int, default=20, help="Number of epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("-do", "--dropout", type=float, default=0.2, help="Dropout for GRU.")
    parser.add_argument("-nl", "--num_layers", type=int, default=4, help="GRU layers.")
    parser.add_argument("-hs", "--hidden_size", type=int, default=128, help="GRU hidden size.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
