import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch

from dataset import fetch_stock_data, fill_missing_dates, preprocess_data

start_date = "2020-01-01"
end_date = "2024-01-01"

#companies
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

df = fetch_stock_data(tickers, start_date, end_date)
df = preprocess_data(df)
df = fill_missing_dates(df)
df = preprocess_data(df)

max_prediction_length = 5
max_encoder_length = 90
training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close",
    group_ids=["Ticker"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["Ticker"],  #static feature (per stock)
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["Open", "High", "Low", "Close", "Volume", "Market Cap",
                                "PE Ratio", "PB Ratio", "EPS", "Debt-to-Equity"],  #features
    target_normalizer=GroupNormalizer(groups=["Ticker"], transformation="softplus"),  #normalize per stock
    allow_missing_timesteps=True,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True
)

batch_size = 32
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

def main_train():

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.003, #identified w/ pl lr finder
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = trainer.checkpoint_callback.best_model_path

def main_eval(model_path):
    tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
    predictions = tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
    MAE()(predictions.output, predictions.y)

    raw_predictions = tft.predict(
        val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
    )

    for idx in range(len(tickers)):
        tft.plot_prediction(
            raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
        )

if '__main__' == __name__:
    best_model = main_train()
    #main_eval(best_model)
