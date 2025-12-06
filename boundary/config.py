"""
Configuration file.
"""

import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

# Allow overriding CUDA use via env flag USE_CUDA=0/1 to help debugging on CPU.
USE_CUDA = torch.cuda.is_available() and os.environ.get("USE_CUDA", "1") not in ("0", "false", "False")
DEVICES = [int(os.environ.get("CUDA_DEVICE_INDEX", 0))]
CUDA_DEVICE = torch.device(f"cuda:{DEVICES[0]}") if USE_CUDA else torch.device("cpu")

LEARNING_RATE = 1e-2

BATCH_SIZE = 128
NUM_WORKERS = 20


# Paths/settings for saving/loading precomputed Palmtree embeddings.
EMBEDDING_DIR = os.environ.get("EMBEDDING_DIR", str(BASE_DIR / "data" / "palmtree_embeddings"))
# Number of binary (pickle) files per shard when computing embeddings.
EMBEDDING_SHARD_SIZE = int(os.environ.get("EMBEDDING_SHARD_SIZE", 100))
