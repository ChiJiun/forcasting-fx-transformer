import time
from functools import partial

import numpy as np
import torch
from gluonts.time_feature import time_features_from_frequency_str
from transformers import (
    AutoformerConfig,
    AutoformerForPrediction,
    InformerConfig,
    InformerForPrediction,
    PatchTSTConfig,
    PatchTSTForPrediction,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
)

import PatchTST
import autoformer
import informer
import transformer


def _batch_get(batch, name):
    return batch[name] if name in batch else None


def _patchtst_values(values):
    values = values.float()
    return values.unsqueeze(-1) if values.ndim == 2 else values


def run_sequence_model(name, module, config_cls, model_cls, lags_sequence):
    freq = "1D"
    prediction_length = 30
    time_features = time_features_from_frequency_str(freq)
    dataset = module.prepare_dataset()

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    for split in (train_dataset, validation_dataset, test_dataset):
        split.set_transform(partial(module.transform_start_field, freq=freq))

    config = config_cls(
        prediction_length=prediction_length,
        context_length=prediction_length * 3,
        lags_sequence=lags_sequence,
        num_time_features=len(time_features) + 1,
        num_static_categorical_features=1,
        num_dynamic_real_features=3,
        cardinality=[len(train_dataset)],
        embedding_dimension=[1],
        encoder_layers=4,
        decoder_layers=4,
        d_model=32,
    )
    model = model_cls(config)
    train_loader = module.create_train_dataloader(
        config, freq, train_dataset, batch_size=4, num_batches_per_epoch=1
    )
    validation_loader = module.create_train_dataloader(
        config, freq, validation_dataset, batch_size=4, num_batches_per_epoch=1
    )
    test_loader = module.create_backtest_dataloader(config, freq, test_dataset, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    start = time.time()
    model.train()
    batch = next(iter(train_loader))
    optimizer.zero_grad()
    outputs = model(
        static_categorical_features=_batch_get(batch, "static_categorical_features"),
        past_time_features=batch["past_time_features"].float(),
        past_values=batch["past_values"],
        future_time_features=batch["future_time_features"].float(),
        future_values=batch["future_values"],
        past_observed_mask=batch["past_observed_mask"],
        future_observed_mask=batch["future_observed_mask"],
    )
    train_loss = float(outputs.loss.detach())
    outputs.loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        batch = next(iter(validation_loader))
        outputs = model(
            static_categorical_features=_batch_get(batch, "static_categorical_features"),
            past_time_features=batch["past_time_features"].float(),
            past_values=batch["past_values"],
            future_time_features=batch["future_time_features"].float(),
            future_values=batch["future_values"],
            past_observed_mask=batch["past_observed_mask"],
            future_observed_mask=batch["future_observed_mask"],
        )
        validation_loss = float(outputs.loss.detach())

        batch = next(iter(test_loader))
        generated = model.generate(
            static_categorical_features=_batch_get(batch, "static_categorical_features"),
            past_time_features=batch["past_time_features"].float(),
            past_values=batch["past_values"],
            future_time_features=batch["future_time_features"].float(),
            past_observed_mask=batch["past_observed_mask"],
        )

    print(
        f"{name}: train_loss={train_loss:.4f} "
        f"validation_loss={validation_loss:.4f} "
        f"forecast_shape={tuple(generated.sequences.shape)} "
        f"seconds={time.time() - start:.1f}"
    )


def run_patchtst():
    freq = "1D"
    prediction_length = 30
    dataset = PatchTST.prepare_dataset()
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    for split in (train_dataset, validation_dataset, test_dataset):
        split.set_transform(partial(PatchTST.transform_start_field, freq=freq))

    config = PatchTSTConfig(
        prediction_length=prediction_length,
        context_length=prediction_length * 3,
        num_hidden_layers=2,
        d_model=16,
        num_input_channels=1,
    )
    model = PatchTSTForPrediction(config)
    train_loader = PatchTST.create_train_dataloader(
        config, freq, train_dataset, batch_size=4, num_batches_per_epoch=1
    )
    validation_loader = PatchTST.create_train_dataloader(
        config, freq, validation_dataset, batch_size=4, num_batches_per_epoch=1
    )
    test_loader = PatchTST.create_test_dataloader(config, freq, test_dataset, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    start = time.time()
    model.train()
    batch = next(iter(train_loader))
    optimizer.zero_grad()
    outputs = model(
        past_values=_patchtst_values(batch["past_values"]),
        future_values=_patchtst_values(batch["future_values"]),
    )
    train_loss = float(outputs.loss.detach())
    outputs.loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        batch = next(iter(validation_loader))
        outputs = model(
            past_values=_patchtst_values(batch["past_values"]),
            future_values=_patchtst_values(batch["future_values"]),
        )
        validation_loss = float(outputs.loss.detach())

        batch = next(iter(test_loader))
        generated = model.generate(past_values=_patchtst_values(batch["past_values"]))

    print(
        f"patchtst: train_loss={train_loss:.4f} "
        f"validation_loss={validation_loss:.4f} "
        f"forecast_shape={tuple(generated.sequences.shape)} "
        f"seconds={time.time() - start:.1f}"
    )


if __name__ == "__main__":
    transformer_lags = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]
    autoformer_lags = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 30, 31, 56, 84]
    run_sequence_model(
        "transformer",
        transformer,
        TimeSeriesTransformerConfig,
        TimeSeriesTransformerForPrediction,
        transformer_lags,
    )
    run_sequence_model("informer", informer, InformerConfig, InformerForPrediction, autoformer_lags)
    run_sequence_model("autoformer", autoformer, AutoformerConfig, AutoformerForPrediction, autoformer_lags)
    run_patchtst()
