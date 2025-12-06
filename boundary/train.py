from multiprocessing.spawn import freeze_support

import glob
from torch.utils.data import TensorDataset, DataLoader, random_split
from model import NN
from config import *

device = CUDA_DEVICE


def _load_embeddings():
    shard_paths = []
    if os.path.isdir(EMBEDDING_DIR):
        shard_paths = sorted(glob.glob(os.path.join(EMBEDDING_DIR, "*.pt")))

    if not shard_paths:
        raise FileNotFoundError(
            f"No embedding shards found. Generate them with create_embeddings.py "
            f"into {EMBEDDING_DIR} or set EMBEDDING_CACHE_PATH to a valid file."
        )

    embeddings_list = []
    labels_list = []
    for path in shard_paths:
        cached = torch.load(path, map_location="cpu")
        embeddings_list.append(cached["embeddings"])
        labels_list.append(cached["labels"])

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    print(f"Loaded {len(shard_paths)} embedding shard(s) totaling {len(labels)} samples")
    return embeddings, labels


class Classifier(object):
    def __init__(self):
        self.mode = "train"
        self.device = device
        effective_workers = max(0, min(NUM_WORKERS, os.cpu_count() or 1, 8))
        embeddings, labels = _load_embeddings()
        dataset = TensorDataset(embeddings, labels)
        val_len = max(1, int(0.1 * len(dataset)))
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
        self.dataloaders = {
            "train": DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=effective_workers,
                                shuffle=True, pin_memory=True),
            "val": DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=effective_workers,
                              shuffle=False, pin_memory=True),
        }
        print(f"Loaded cached embeddings | train {len(train_ds)} | val {len(val_ds)}")
        self.model = NN(128, 20, 2, device)
        self.model = self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=10,
                                                                    min_lr=1e-6)
        scaler_cls = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
        self.scaler = scaler_cls(enabled=USE_CUDA)

    def iterate(self, phase, dataloader):
        total_loss = 0.0
        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None

        for batch_idx, batch in enumerate(dataloader):
            embeddings, labels = batch
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(phase == "train"):
                embeddings = embeddings.to(self.device, non_blocking=True, dtype=torch.float32)
                with torch.amp.autocast("cuda", enabled=USE_CUDA):
                    logits = self.model(embeddings)
                    logits_flat = logits.reshape(-1, logits.shape[-1])
                    labels_flat = labels.view(-1)
                    batch_loss = self.criterion(logits_flat, labels_flat)

                if phase == 'train':
                    self.scaler.scale(batch_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            loss_value = float(batch_loss.detach().item()) if isinstance(batch_loss, torch.Tensor) else float(
                batch_loss)
            total_loss += loss_value
            running_loss = total_loss / (batch_idx + 1)

            preds = logits.detach().argmax(dim=-1).cpu()
            labels_cpu = labels.detach().cpu()
            preds_flat = preds.view(-1)
            labels_flat = labels_cpu.view(-1)
            batch_TP = int(((labels_flat == 1) & (preds_flat == 1)).sum().item())
            batch_TN = int(((labels_flat == 0) & (preds_flat == 0)).sum().item())
            batch_FN = int(((labels_flat == 1) & (preds_flat == 0)).sum().item())
            batch_FP = int(((labels_flat == 0) & (preds_flat == 1)).sum().item())

            p = batch_TP / (batch_TP + batch_FP + 0.0001)
            r = batch_TP / (batch_TP + batch_FN + 0.0001)
            batch_F1 = 2 * r * p / (r + p + 0.0001)
            batch_acc = (batch_TP + batch_TN) / (batch_TP + batch_TN + batch_FP + batch_FN)

            if batch_idx % 10 == 0:
                progress = f"[{batch_idx + 1}/{total_batches}]" if total_batches else f"[{batch_idx + 1}]"
                print(
                    f"{progress} {phase} acc:{batch_acc:.4f}, F1:{batch_F1:.4f}, loss:{loss_value:.4f}, running_loss:{running_loss:.4f}")
        return total_loss

    def train(self):
        train_dataloader = self.dataloaders['train']
        val_dataloader = self.dataloaders['val']
        print("# batches of sequence for training", len(train_dataloader))
        print("# batches of sequence for validation", len(val_dataloader))
        for epoch in range(1, 21):
            print(f"Epoch {epoch}")
            self.model.train()
            train_loss = self.iterate('train', train_dataloader)

            self.model.eval()
            with torch.no_grad():
                val_loss = self.iterate('val', val_dataloader)
            self.scheduler.step(val_loss)
            avg_train_loss = train_loss / max(1, len(train_dataloader))
            avg_val_loss = val_loss / max(1, len(val_dataloader))
            print(f"Epoch {epoch} complete | avg train loss {avg_train_loss:.4f} | avg val loss {avg_val_loss:.4f}")

    def save_model(self):
        save_dir = BASE_DIR / "trained_model"
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / "lstm_classifier.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hidden_size": self.model.hidden_size,
            "sequence_len": self.model.sequence_len,
            "num_classes": self.model.num_classes,
        }, str(save_path))
        print(f"Saved trained LSTM to {save_path}")


if __name__ == "__main__":
    freeze_support()
    classifier = Classifier()
    classifier.train()
    classifier.save_model()
