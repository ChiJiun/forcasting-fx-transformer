import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import get_seasonality, time_features_from_frequency_str
from transformers import InformerConfig, InformerForPrediction

import informer


FREQ = "1D"
LAGS_SEQUENCE = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 30, 31, 56, 84]
GRID = [
    {"prediction_length": 14, "context_length": 60},
    {"prediction_length": 14, "context_length": 90},
    {"prediction_length": 14, "context_length": 120},
    {"prediction_length": 30, "context_length": 60},
    {"prediction_length": 30, "context_length": 90},
    {"prediction_length": 30, "context_length": 120},
]
TRAINING_CONFIG = {
    "learning_rate": 4e-4,
    "epochs": 120,
    "batch_size": 128,
    "num_batches_per_epoch": 200,
    "train_num_instances": 5.0,
    "seed": 42,
}


def compute_metrics(forecast_median, test_dataset, prediction_length):
    seasonality = get_seasonality(FREQ)
    rows = []
    for item_id, time_series in enumerate(test_dataset):
        training_data = np.asarray(time_series["target"][:-prediction_length])
        ground_truth = np.asarray(time_series["target"][-prediction_length:])
        predictions = forecast_median[item_id]

        if len(training_data) > seasonality:
            scale = np.mean(np.abs(training_data[seasonality:] - training_data[:-seasonality]))
        else:
            scale = np.mean(np.abs(np.diff(training_data)))

        denominator = (np.abs(predictions) + np.abs(ground_truth)) / 2
        rows.append({
            "item_id": time_series["item_id"],
            "mase": np.mean(np.abs(predictions - ground_truth)) / scale,
            "smape": np.mean(np.where(denominator == 0, 0, np.abs(predictions - ground_truth) / denominator)),
            "mse": np.mean((predictions - ground_truth) ** 2),
            "mae": np.mean(np.abs(predictions - ground_truth)),
        })

    return pd.DataFrame(rows)


def run_experiment(grid_config, base_output_dir):
    prediction_length = grid_config["prediction_length"]
    context_length = grid_config["context_length"]
    run_name = f"pred{prediction_length}_ctx{context_length}"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metric.csv").exists() and (output_dir / "training_log.csv").exists():
        print(f"{run_name} already completed, loading existing metrics", flush=True)
        metrics = pd.read_csv(output_dir / "metric.csv")
        training_log = pd.read_csv(output_dir / "training_log.csv")
        summary = {
            "run": run_name,
            "prediction_length": prediction_length,
            "context_length": context_length,
            "best_validation_loss": float(training_log["validation_loss"].min()),
        }
        for metric_name in ["mase", "smape", "mse", "mae"]:
            summary[f"mean_{metric_name}"] = float(metrics[metric_name].mean())
            summary[f"median_{metric_name}"] = float(metrics[metric_name].median())
        return summary

    torch.manual_seed(TRAINING_CONFIG["seed"])
    np.random.seed(TRAINING_CONFIG["seed"])

    time_features = time_features_from_frequency_str(FREQ)
    dataset = informer.prepare_dataset()
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    for split in (train_dataset, validation_dataset, test_dataset):
        split.set_transform(partial(informer.transform_start_field, freq=FREQ))

    config = InformerConfig(
        prediction_length=prediction_length,
        context_length=context_length,
        lags_sequence=LAGS_SEQUENCE,
        num_time_features=len(time_features) + 1,
        num_static_categorical_features=1,
        num_dynamic_real_features=3,
        cardinality=[len(train_dataset)],
        embedding_dimension=[1],
        encoder_layers=4,
        decoder_layers=4,
        d_model=32,
    )

    with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "model": "informer",
                "grid_config": grid_config,
                "training_config": TRAINING_CONFIG,
                "lags_sequence": LAGS_SEQUENCE,
            },
            config_file,
            indent=2,
        )

    model = InformerForPrediction(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=1e-1,
    )

    train_dataloader = informer.create_train_dataloader(
        config=config,
        freq=FREQ,
        data=train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        num_batches_per_epoch=TRAINING_CONFIG["num_batches_per_epoch"],
        train_num_instances=TRAINING_CONFIG["train_num_instances"],
    )
    validation_dataloader = informer.create_train_dataloader(
        config=config,
        freq=FREQ,
        data=validation_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        num_batches_per_epoch=TRAINING_CONFIG["num_batches_per_epoch"],
    )
    test_dataloader = informer.create_backtest_dataloader(
        config=config,
        freq=FREQ,
        data=test_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
    )

    training_logs = []
    best_validation_loss = float("inf")
    for epoch in range(TRAINING_CONFIG["epochs"]):
        epoch_start = time.time()
        model.train()
        train_losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch["static_categorical_features"],
                past_time_features=batch["past_time_features"].float(),
                past_values=batch["past_values"],
                future_time_features=batch["future_time_features"].float(),
                future_values=batch["future_values"],
                past_observed_mask=batch["past_observed_mask"],
                future_observed_mask=batch["future_observed_mask"],
            )
            outputs.loss.backward()
            optimizer.step()
            train_losses.append(float(outputs.loss.detach()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_dataloader:
                outputs = model(
                    static_categorical_features=batch["static_categorical_features"],
                    past_time_features=batch["past_time_features"].float(),
                    past_values=batch["past_values"],
                    future_time_features=batch["future_time_features"].float(),
                    future_values=batch["future_values"],
                    past_observed_mask=batch["past_observed_mask"],
                    future_observed_mask=batch["future_observed_mask"],
                )
                validation_losses.append(float(outputs.loss.detach()))

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        training_logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "time_seconds": time.time() - epoch_start,
            "is_best": is_best,
        })
        pd.DataFrame(training_logs).to_csv(output_dir / "training_log.csv", index=False)
        print(
            f"{run_name} epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}",
            flush=True,
        )

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    forecasts = []
    with torch.no_grad():
        for batch in test_dataloader:
            outputs = model.generate(
                static_categorical_features=batch["static_categorical_features"],
                past_time_features=batch["past_time_features"].float(),
                past_values=batch["past_values"],
                future_time_features=batch["future_time_features"].float(),
                past_observed_mask=batch["past_observed_mask"],
            )
            forecasts.append(outputs.sequences.cpu().numpy())

    forecasts = np.vstack(forecasts)
    forecast_median = np.median(forecasts, axis=1)
    metrics = compute_metrics(forecast_median, test_dataset, prediction_length)
    metrics.to_csv(output_dir / "metric.csv", index=False)

    summary = {
        "run": run_name,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "best_validation_loss": best_validation_loss,
    }
    for metric_name in ["mase", "smape", "mse", "mae"]:
        summary[f"mean_{metric_name}"] = float(metrics[metric_name].mean())
        summary[f"median_{metric_name}"] = float(metrics[metric_name].median())
    return summary


def main():
    output_dir = Path("outputs") / "informer_grid_full"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for grid_config in GRID:
        summaries.append(run_experiment(grid_config, output_dir))
        pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)

    summary_df = pd.DataFrame(summaries).sort_values("mean_mase")
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print("\nSummary sorted by mean_mase")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
