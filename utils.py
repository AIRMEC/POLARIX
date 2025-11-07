import os
import numpy as np
import random
import h5py
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    SequentialSampler,
    WeightedRandomSampler,
)

from sklearn.utils.class_weight import compute_sample_weight

from POLARIX import INPUT_FEATURE_SIZE


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate(batch):
    # note if return things that don't need to be trained on, there is no need to call torch tensor, keep a numpy array.
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    slide_id = [item[2] for item in batch]

    return [img, label, slide_id]


def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def evaluate_loader(model, device, loader, loss_fn=None):
    """
    Runs a full pass of the model over a loader and returns loss and predictions.
    """
    model.eval()
    loss_fn = loss_fn or nn.BCEWithLogitsLoss()

    eval_loss = 0.0
    probs = np.zeros(len(loader))
    logits = np.zeros(len(loader))
    labels = np.zeros(len(loader), dtype=int)
    slide_ids = []

    with torch.inference_mode():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device), label.to(device).float()
            slide_id = slide_id[0]

            logit, Y_prob, _, _ = model(data)
            logit = logit.squeeze(dim=1)
            Y_prob = Y_prob.squeeze(dim=1)

            loss = loss_fn(logit, label)
            eval_loss += loss.item()

            probs[batch_idx] = Y_prob.cpu().item()
            logits[batch_idx] = logit.cpu().item()
            labels[batch_idx] = label.cpu().item()
            slide_ids.append(slide_id)

    eval_loss /= len(loader)

    return eval_loss, probs, logits, labels, slide_ids


def get_val_loader(val_split, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    val_loader = DataLoader(
        dataset=val_split,
        batch_size=1,  # model expects one bag of features at the time.
        sampler=SequentialSampler(val_split),
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return val_loader


def get_train_loader(train_split, method, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    if method == "random":
        print("random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            shuffle=True,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    elif method == "balanced":
        print("balanced sampling setting")
        train_labels = train_split.slide_df["label"]

        # Compute sample weights to alleviate class imbalance with weighted sampling.
        sample_weights = compute_sample_weight("balanced", train_labels)

        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            # Use the weighted sampler using the precomputed sample weights.
            # Note that replacement is true by default, so
            # some slides of rare classes will be sampled multiple times per epoch.
            shuffle=False,
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    return train_loader


class FeatureBags(Dataset):
    def __init__(self, df, data_dir):
        self.slide_df = df.copy().reset_index(drop=True)
        self.data_dir = data_dir

    def _get_feature_path(self, slide_id):
        return os.path.join(self.data_dir, f"{slide_id}_features.h5")

    def __getitem__(self, idx):
        slide_id = self.slide_df["slide_id"][idx]
        label = self.slide_df["label"][idx]

        full_path = self._get_feature_path(slide_id)
        with h5py.File(full_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            # coords = hdf5_file["coords"][:]

        assert (
            features.shape[1] == INPUT_FEATURE_SIZE
        ), f"Expected feature dim {INPUT_FEATURE_SIZE}, got {features.shape[1]}"
        features = torch.from_numpy(features)

        return features, label, slide_id

    def __len__(self):
        return len(self.slide_df)
