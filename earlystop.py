import numpy as np
import pandas as pd
import os
import torch
import joblib


class MonitorBestModelEarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience and saves the best model.
    """

    def __init__(
        self,
        patience=15,
        min_epochs=20,
        saving_checkpoint=True,
        hpset=None,
        output_dir=None,
    ):
        """
        Initializes the early stopping monitor.
        Args:
            patience (int): How long to wait after the last time validation loss improved. Default: 15
            min_epochs (int): Minimum number of epochs to wait before considering early stopping. Default: 20
            saving_checkpoint (bool): If True, saves the model checkpoint when validation loss improves. Default: True
            hpset (int): Hyperparameter set identifier.
            output_dir (str): Directory to save checkpoints and predictions.
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.early_stop = False

        self.eval_loss_min = np.inf
        self.best_loss_score = None
        self.best_epoch_loss = None

        self.best_opt_metric_score = 0.0  # model will be optimized on this metric
        self.best_epoch = None

        self.saving_checkpoint = saving_checkpoint

        self.hpset = hpset
        self.output_dir = os.path.abspath(output_dir) if output_dir else None

        if self.saving_checkpoint and self.output_dir is None:
            raise ValueError(
                "output_dir must be provided when saving_checkpoint is enabled."
            )

    def __call__(
        self, epoch, eval_loss, eval_opt_metric, model, platt_model, preds, preds_train
    ):
        """
        Checks if training should be stopped based on validation loss and saves the best model.
        """
        loss_score = -eval_loss
        opt_score = eval_opt_metric  # AUC

        if self.best_loss_score is None:
            self._update_loss_scores(loss_score, eval_loss, epoch)
            self._update_metrics_scores(opt_score, epoch)

        # If validation loss starts increasing, begin counting for early stopping
        elif loss_score < self.best_loss_score:
            self.counter += 1
            print(
                f"Evaluation loss does not decrease : Starting Early stopping counter {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True

        # If validation loss is still decreasing, reset the counter and save the model
        else:
            print(
                f"Epoch {epoch} validation loss decreased ({self.eval_loss_min:.6f} --> {eval_loss:.6f})"
            )
            self._update_loss_scores(loss_score, eval_loss, epoch)
            self._update_metrics_scores(opt_score, epoch)
            self.save_checkpoint_predictions(model, platt_model, preds, preds_train)
            self.counter = 0

        # If the optimization metric (e.g., AUC) improves, update scores and save the model
        if opt_score > self.best_opt_metric_score:
            print(
                f"Epoch {epoch}: AUC improved ({self.best_opt_metric_score:.4f} --> {opt_score:.4f})"
            )
            self._update_metrics_scores(opt_score, epoch)
            self._update_loss_scores(loss_score, eval_loss, epoch)
            self.save_checkpoint_predictions(model, platt_model, preds, preds_train)

    def save_checkpoint_predictions(self, model, platt_model, preds, preds_train):
        """
        Saves the model checkpoint, Platt scaler, and predictions.
        """
        if not self.saving_checkpoint or self.output_dir is None:
            return

        print(f"Saving checkpoint and predictions")
        base_name = "polarix"

        # Save model checkpoint
        checkpoints_dir = os.path.join(
            self.output_dir, "checkpoints", "final", base_name
        )
        os.makedirs(checkpoints_dir, exist_ok=True)
        filepath_check = os.path.join(
            checkpoints_dir,
            f"{base_name}_hp{self.hpset}_checkpoint.pt",
        )
        torch.save(model.state_dict(), filepath_check)

        # Save Platt scaler model and coefficients
        platt_dir = checkpoints_dir
        filepath_platt = os.path.join(
            platt_dir,
            f"{base_name}_hp{self.hpset}_PlattScaler.pkl",
        )
        joblib.dump(platt_model, filepath_platt)
        alpha = platt_model.coef_[0][0]
        beta = platt_model.intercept_[0]
        coefficients_df = pd.DataFrame({"alpha": [alpha], "beta": [beta]})
        filepath_platt_coef = os.path.join(
            platt_dir,
            f"{base_name}_hp{self.hpset}_PlattScalerCOEF.csv",
        )
        coefficients_df.to_csv(filepath_platt_coef, index=False)

        # Save evaluation predictions
        predictions_dir = os.path.join(
            self.output_dir, "predictions", "final", base_name
        )
        os.makedirs(predictions_dir, exist_ok=True)
        filepath_pred = os.path.join(
            predictions_dir,
            f"{base_name}_hp{self.hpset}_predictions.csv",
        )
        preds.to_csv(filepath_pred, index=False)

        # Save training predictions
        filepath_pred_train = os.path.join(
            predictions_dir,
            f"{base_name}_hp{self.hpset}_predictions_TRAIN.csv",
        )
        preds_train.to_csv(filepath_pred_train, index=False)

    def _update_loss_scores(self, loss_score, eval_loss, epoch):
        self.eval_loss_min = eval_loss
        self.best_loss_score = loss_score
        self.best_epoch_loss = epoch
        print(
            f"Updating loss at epoch {self.best_epoch_loss} -> {self.eval_loss_min:.6f}"
        )

    def _update_metrics_scores(self, opt_score, epoch):
        self.best_opt_metric_score = opt_score
        self.best_epoch = epoch
        print(
            f"Updating Opt metric at epoch {self.best_epoch} -> {self.best_opt_metric_score:.6f}"
        )
