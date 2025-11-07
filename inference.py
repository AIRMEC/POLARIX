import time
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import joblib

from POLARIX import POLARIX, INPUT_FEATURE_SIZE
from utils import *


def main(args):
    set_seed()

    start_time = time.time()

    df_test = pd.read_csv(args.manifest_test)
    print(f"Read {args.manifest_test} dataset containing {len(df_test)} samples")

    print("Initializing POLARIX model")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using fixed input feature size: {INPUT_FEATURE_SIZE}")
    model = POLARIX().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint_POLARIX_model, map_location=device), strict=True
    )
    print(f"Finished loading POLARIX in {(time.time() - start_time):.2f}s")

    model.eval()

    dataset = FeatureBags(
        df=df_test,
        data_dir=args.data_features_dir,
    )

    loader = get_val_loader(val_split=dataset, workers=args.workers)

    eval_loss, probs, logits, labels, slide_ids = evaluate_loader(model, device, loader)
    eval_auc = roc_auc_score(labels, probs)
    print(f"Eval loss: {eval_loss}, AUC: {eval_auc:.4f}")

    print(f"Eval AUC {eval_auc}")

    # Apply Platt scaling for probability calibration
    platt_scaler = joblib.load(args.checkpoint_platt_model)
    calprobs = platt_scaler.predict_proba(logits.reshape(-1, 1))[:, 1]

    preds_df = pd.DataFrame(probs, columns=[f"prob"])
    preds_df["prob calibrated"] = calprobs
    preds_df["logit"] = logits
    preds_df["label"] = labels
    preds_df["slide_id"] = slide_ids

    # Save into a csv file.
    print(f"Saving predictions...")
    preds_df.to_csv("predictions.csv")

    print(f"Finished making POLARIX predictions in {(time.time() - start_time):.2f}s")

    total_time = time.time() - start_time
    avg_time_per_slide = total_time / len(df_test)
    print(f"Average prediction time per slide: {avg_time_per_slide:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--manifest_test",
        type=str,
        help="CSV file of test set listing all work_ids, slides, and labels",
    )
    parser.add_argument(
        "--checkpoint_POLARIX_model",
        type=str,
        help="path to POLARIX model checkpoint",
    )
    parser.add_argument(
        "--checkpoint_platt_model",
        type=str,
        help="path to Platt model checkpoint",
    )
    parser.add_argument(
        "--data_features_dir",
        type=str,
        help="Directory where all *_features.h5 files are stored",
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    main(args)
