import glob
import os
import pickle
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from binary_file_format import BinaryFileDef  # noqa: E402
from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    EMBEDDING_DIR,
    EMBEDDING_SHARD_SIZE,
    BASE_DIR,
)
import eval_utils as utils


def build_samples_from_pickle(path: str) -> List[Tuple[List[str], List[int]]]:
    samples = []
    with open(path, "rb") as f:
        binary: BinaryFileDef = pickle.load(f)
        for func in binary.functions:
            randomint = random.randint(5, 15)
            begin = func.inst_strings[:20 - randomint]
            end = func.inst_strings[-randomint:]
            if len(begin) + len(end) == 20:
                sample = end + begin
                if len(sample) != 20:
                    continue
                label = [0 for _ in range(20)]
                label[randomint] = 1
                samples.append((sample, label))
    return samples


def create_embeddings_from_pickles(pickle_glob: str = None,
                                   cache_dir: str = EMBEDDING_DIR,
                                   shard_size_files: int = EMBEDDING_SHARD_SIZE):
    os.makedirs(cache_dir, exist_ok=True)
    if pickle_glob is None:
        pickle_glob = str(BASE_DIR.parent / "data" / "pickles" / "**" / "*.pkl")
    pickle_files = sorted(glob.glob(pickle_glob, recursive=True))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found matching {pickle_glob}")

    palmtree = utils.UsableTransformer(
        model_path=str(BASE_DIR / "palmtree" / "transformer.ep19"),
        vocab_path=str(BASE_DIR / "palmtree" / "vocab"),
    )
    shard_idx = 0

    for start in range(0, len(pickle_files), shard_size_files):
        shard_files = pickle_files[start:start + shard_size_files]
        sequences = []
        labels = []

        for pk in shard_files:
            for seq, lbl in build_samples_from_pickle(pk):
                sequences.append(seq)
                labels.append(torch.tensor(lbl, dtype=torch.long))

        # Batch through Palmtree encoder to avoid OOM.
        all_embeddings = []
        all_labels = []
        dataloader = DataLoader(
            list(zip(sequences, labels)),
            batch_size=BATCH_SIZE,
            num_workers=max(0, min(NUM_WORKERS, os.cpu_count() or 1, 8)),
            shuffle=False,
            pin_memory=True,
        )

        with torch.no_grad():
            for batch_sequences, batch_labels in dataloader:
                embs = palmtree.encode(batch_sequences)
                # Fix layout if Palmtree returned seq-first.
                if embs.ndim == 3 and embs.shape[0] != batch_labels.shape[0] and embs.shape[1] == batch_labels.shape[0]:
                    embs = embs.permute(1, 0, 2).contiguous()
                if embs.shape[0] != batch_labels.shape[0]:
                    raise RuntimeError(
                        f"Embedding batch size {embs.shape[0]} does not match labels {batch_labels.shape[0]} "
                        f"for shard starting at {start}"
                    )
                all_embeddings.append(embs.cpu())
                all_labels.append(batch_labels.cpu())

        # All batches should now be batch-first and consistent; pad only if hidden differs.
        hidden_sizes = [e.shape[-1] for e in all_embeddings]
        target_hidden = max(hidden_sizes)
        if len(set(hidden_sizes)) > 1:
            padded_embeddings = []
            for e in all_embeddings:
                if e.shape[-1] < target_hidden:
                    pad_width = target_hidden - e.shape[-1]
                    e = F.pad(e, (0, pad_width))
                padded_embeddings.append(e)
            embeddings = torch.cat(padded_embeddings, dim=0)
        else:
            embeddings = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        if embeddings.shape[0] != labels_tensor.shape[0]:
            raise RuntimeError(
                f"Final embedding count {embeddings.shape[0]} does not match labels {labels_tensor.shape[0]} "
                f"in shard {shard_idx}"
            )

        shard_path = os.path.join(cache_dir, f"palmtree_embeddings_shard_{shard_idx:04d}.pt")
        torch.save({"embeddings": embeddings, "labels": labels_tensor, "source_files": shard_files}, shard_path)
        print(f"Saved shard {shard_idx} from {len(shard_files)} pickles to {shard_path} | "
              f"embeddings shape {embeddings.shape} | labels shape {labels_tensor.shape}")
        shard_idx += 1


if __name__ == "__main__":
    create_embeddings_from_pickles()
