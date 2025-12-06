import random
import json
from torch.utils.data import DataLoader
import torch
import os


def collate_wrapper(batch):
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return sequences, labels


def split(Samples, train_val_split=0.1):
    Samples = Samples[:10000]
    random.shuffle(Samples)

    cross_validation_val = Samples[:int(train_val_split * len(Samples))]
    cross_validation_train = Samples[int(train_val_split * len(Samples)):]
    return cross_validation_train, cross_validation_val


def generator(jsons, batch_size, num_workers, max_files=None, max_samples=None):
    # Cap workers to avoid hitting OS file descriptor limits
    num_workers = max(0, min(num_workers, os.cpu_count() or 1, 8))
    Sequences = []
    file_slice = jsons if max_files is None else jsons[:max_files]
    for f_name in file_slice:
        with open(f_name) as inf:
            for line in inf:
                s_info = json.loads(line.strip())
                sequence = s_info[0]
                label = torch.tensor(s_info[1])
                Sequences.append([sequence, label])
                if max_samples is not None and len(Sequences) >= max_samples:
                    break
        if max_samples is not None and len(Sequences) >= max_samples:
            break
    cross_validation_train, cross_validation_val = split(Sequences)
    train_dataloader = DataLoader(cross_validation_train,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate_wrapper,
                                  pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(cross_validation_val,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=collate_wrapper,
                                pin_memory=True, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    return dataloaders
