import os
import subprocess
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from POLARIX import POLARIX, INPUT_FEATURE_SIZE
from utils import *
from earlystop import MonitorBestModelEarlyStopping


def evaluate_model(epoch, model, device, loader, writer):
    eval_loss, probs, logits, labels, slide_ids = evaluate_loader(model, device, loader)

    auc = roc_auc_score(labels, probs)

    print(f"Eval epoch: {epoch}, loss: {eval_loss}, AUC: {auc:.4f}")
    writer.add_scalar("Loss/eval", eval_loss, epoch)
    writer.add_scalar("auc/eval", auc, epoch)

    return eval_loss, auc, probs, logits, labels, slide_ids


def train_one_epoch(epoch, model, device, train_loader, optimizer, writer, loss_fn):
    model.train()
    epoch_start_time = time.time()
    train_loss = 0.0

    probs = np.zeros(len(train_loader))
    logits = np.zeros(len(train_loader))
    labels = np.zeros(len(train_loader))
    slide_ids = []

    batch_start_time = time.time()
    for batch_idx, (data, label, slide_id) in enumerate(train_loader):
        data_load_duration = time.time() - batch_start_time

        data, label = data.to(device), label.to(device).float()
        slide_id = slide_id[0]

        logit, Y_prob, _, _ = model(data)
        logit = logit.squeeze(dim=1)
        Y_prob = Y_prob.squeeze(dim=1)

        loss = loss_fn(logit, label)
        train_loss += loss.item()

        probs[batch_idx] = Y_prob.detach().cpu().item()
        logits[batch_idx] = logit.detach().cpu().item()
        labels[batch_idx] = label.cpu().item()
        slide_ids.append(slide_id)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_duration = time.time() - batch_start_time
        batch_start_time = time.time()

        writer.add_scalar("duration/data_load", data_load_duration, epoch)
        writer.add_scalar("duration/batch", batch_duration, epoch)

    epoch_duration = time.time() - epoch_start_time
    print(f"Finished training on epoch {epoch} in {epoch_duration:.2f}s")

    train_loss /= len(train_loader)

    auc = roc_auc_score(labels, probs)

    print(f"Train epoch: {epoch}, loss: {train_loss}, AUC: {auc:.4f}")

    writer.add_scalar("duration/epoch", epoch_duration, epoch)
    writer.add_scalar("LR", get_lr(optimizer), epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("auc/train", auc, epoch)
    return probs, logits, labels, slide_ids


def run_train_eval_loop(
    train_loader,
    val_loader,
    loss_fn,
    hparams,
    run_id,
    hpset,
    device,
    output_dir,
):
    output_dir = os.path.abspath(output_dir)
    tensorboard_dir = os.path.join(output_dir, "tensorboard", run_id)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    print(f"Using fixed input feature size: {INPUT_FEATURE_SIZE}")
    model = POLARIX(
        precompression_layer=hparams["precompression_layer"],
        feature_size_comp=hparams["feature_size_comp"],
        feature_size_attn=hparams["feature_size_attn"],
        feature_size_comp_post=hparams["feature_size_comp_post"],
        dropout=True,
        p_dropout_fc=hparams["p_dropout_fc"],
        p_dropout_atn=hparams["p_dropout_atn"],
    ).to(device)
    print_model(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["initial_lr"],
        weight_decay=hparams["weight_decay"],
    )

    if hparams["milestones"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hparams["max_epochs"], eta_min=0.0, last_epoch=-1
        )
    else:
        milestones = [int(x) for x in hparams["milestones"].split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=hparams["gamma_lr"]
        )

    monitor_tracker = MonitorBestModelEarlyStopping(
        patience=hparams["earlystop_patience"],
        min_epochs=hparams["earlystop_min_epochs"],
        saving_checkpoint=True,
        hpset=hpset,
        output_dir=output_dir,
    )

    platt_model = LogisticRegression(fit_intercept=True, solver="lbfgs")

    for epoch in range(hparams["max_epochs"]):
        # Train for one epoch
        probs_train, logits_train, labels_train, slide_ids_train = train_one_epoch(
            epoch, model, device, train_loader, optimizer, writer, loss_fn
        )

        # Apply Platt scaling to training predictions
        platt_model.fit(logits_train.reshape(-1, 1), labels_train)
        calprobs_train = platt_model.predict_proba(logits_train.reshape(-1, 1))[:, 1]

        preds_train = pd.DataFrame(probs_train, columns=[f"prob"])
        preds_train["prob calibrated"] = calprobs_train
        preds_train["logit"] = logits_train
        preds_train["label"] = labels_train
        preds_train["slide_id"] = slide_ids_train

        # Evaluate the model
        eval_loss, eval_auc, probs, logits, labels, slide_ids = evaluate_model(
            epoch, model, device, val_loader, writer
        )

        # Apply Platt scaling to validation predictions
        calprobs = platt_model.predict_proba(logits.reshape(-1, 1))[:, 1]

        preds_df = pd.DataFrame(probs, columns=[f"prob"])
        preds_df["prob calibrated"] = calprobs
        preds_df["logit"] = logits
        preds_df["label"] = labels
        preds_df["slide_id"] = slide_ids

        # Check for early stopping
        monitor_tracker(
            epoch, eval_loss, eval_auc, model, platt_model, preds_df, preds_train
        )

        scheduler.step()

        if monitor_tracker.early_stop:
            print(
                f"Early stop criterion reached. Broke off training loop after epoch {epoch}."
            )
            break

    runs_history = {
        "run_id": run_id,
        "best_epoch_loss": monitor_tracker.best_epoch_loss,
        "best_evalLoss": monitor_tracker.eval_loss_min,
        "best_epoch_AUC": monitor_tracker.best_epoch,
        "best_AUC_score": monitor_tracker.best_opt_metric_score,
        **hparams,
    }

    with open(f"../runs_final.txt", "a") as filehandle:
        for _, value in runs_history.items():
            filehandle.write("%s;" % value)
        filehandle.write("\n")

    writer.close()


def main(args):
    set_seed()

    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA device available. Training without CUDA-acceleration is not recommended."
        )

    df = pd.read_csv(args.manifest)
    print(f"Read {args.manifest} dataset containing {len(df)} samples")

    try:
        training_set = df[df["split"] == "train"]
        validation_set = df[df["split"] == "val"]
    except KeyError:
        raise Exception(f"Required column 'split' does not exist in {args.manifest}")

    train_split = FeatureBags(
        df=training_set,
        data_dir=args.data_dir,
    )

    val_split = FeatureBags(
        df=validation_set,
        data_dir=args.data_dir,
    )

    # Generate a unique run ID
    try:
        git_sha = (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8")
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
    ):
        git_sha = "nogit"
    train_run_id = f"{git_sha}-{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"=> Git SHA {train_run_id}")
    print(f"=> Training on {len(train_split)} samples")
    print(f"=> Validating on {len(val_split)} samples")

    # Base hyperparameters
    base_hparams = dict(
        data_dir=os.path.dirname(args.data_dir),
        sampling_method="random",
        max_epochs=100,
        earlystop_patience=30,
        earlystop_min_epochs=30,
        initial_lr=0.00003,
        milestones="5, 15, 25",
        gamma_lr=0.1,
        weight_decay=0.00001,
        # Model architecture parameters. See model class for details.
        precompression_layer=False,
        feature_size_comp=512,
        feature_size_attn=256,
        feature_size_comp_post=128,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
    )

    # Hyperparameter sets for experimentation
    hparam_sets = [
        {**base_hparams},
        {
            **base_hparams,
            "p_dropout_fc": 0.50,
            "p_dropout_atn": 0.50,
            "weight_decay": 0.0001,
            "initial_lr": 0.0001,
            "sampling_method": "balanced",
        },
        {**base_hparams, "class_weighting": True},
        {
            **base_hparams,
            "p_dropout_fc": 0.50,
            "p_dropout_atn": 0.50,
            "weight_decay": 0.0001,
            "initial_lr": 0.0001,
            "class_weighting": True,
        },
    ]

    hps = hparam_sets[args.hp]

    run_id = f"{train_run_id}_hp{args.hp}"
    print(f"Running train-eval loop {args.hp} for {run_id}")
    print(hps)

    train_loader = get_train_loader(
        train_split,
        hps["sampling_method"],
        args.workers,
    )

    val_loader = get_val_loader(val_split, args.workers)

    device = torch.device("cuda")

    # Set up loss function with class weighting if specified
    loss_function = nn.BCEWithLogitsLoss()
    if hps.get("class_weighting", False):
        labels_tensor = torch.tensor(
            train_split.slide_df["label"].values, dtype=torch.float32
        )
        num_pos = int((labels_tensor == 1).sum().item())
        num_neg = int((labels_tensor == 0).sum().item())

        if num_pos == 0 or num_neg == 0:
            raise ValueError(
                "Class weighting requested but only one class is present in the training split."
            )

        pos_weight_value = num_neg / num_pos
        print(f"Using class weighting with pos_weight: {pos_weight_value:.2f}")
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Start the training and evaluation loop
    run_train_eval_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_function,
        hparams=hps,
        run_id=run_id,
        hpset=args.hp,
        device=device,
        output_dir=args.output_dir,
    )
    print("Finished training.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--manifest",
        type=str,
        help="CSV file listing all slides, their labels, and which split (train/test/val) they belong to.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where all *_features.h5 files are stored",
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--hp",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs/final",
        help="Base directory where checkpoints, predictions, and TensorBoard logs are stored.",
    )

    args = parser.parse_args()

    main(args)
