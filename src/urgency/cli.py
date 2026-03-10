"""Command-line interface for the ISIC Lesion Urgency Model.

All commands are documented below and correspond to the stages of the pipeline:
prepare data -> make splits -> train -> calibrate -> evaluate -> infer / serve.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="urgency",
    help="ISIC Lesion Urgency Model CLI. Research use only, not a diagnostic device.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# prepare-isic
# ---------------------------------------------------------------------------


@app.command("prepare-isic")
def prepare_isic(
    input: Path = typer.Option(..., help="Path to ISIC metadata CSV export."),
    output: Path = typer.Option(Path("data"), help="Output directory for labels.csv."),
) -> None:
    """Convert an ISIC metadata export CSV to the standard labels.csv format."""
    from scripts.prepare_isic import convert_isic_export

    convert_isic_export(input_path=input, output_dir=output)


# ---------------------------------------------------------------------------
# make-splits
# ---------------------------------------------------------------------------


@app.command("make-splits")
def make_splits_cmd(
    labels: Path = typer.Option(Path("data/labels.csv"), help="Path to labels.csv."),
    out: Path = typer.Option(Path("data/splits"), help="Directory to save split CSVs."),
    val_frac: float = typer.Option(0.15, help="Fraction for validation split."),
    test_frac: float = typer.Option(0.15, help="Fraction for test split."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Create patient-level train/val/test splits from labels.csv."""
    import pandas as pd

    from urgency.config import UrgencyMappingConfig
    from urgency.data.splits import apply_urgency_mapping, make_splits
    from urgency.utils.checks import check_labels_csv

    check_labels_csv(labels)

    df = pd.read_csv(labels)
    # Apply default urgency mapping before splitting
    mapping = UrgencyMappingConfig()
    df = apply_urgency_mapping(df, mapping)

    splits = make_splits(df, splits_dir=out, val_frac=val_frac, test_frac=test_frac, seed=seed)
    typer.echo(f"Splits saved to {out}")
    for name, split_df in splits.items():
        typer.echo(f"  {name}: {len(split_df)} samples")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command("train")
def train_cmd(
    config: Path = typer.Option(Path("configs/default.yaml"), help="Path to Hydra config YAML."),
    overrides: Optional[list[str]] = typer.Argument(default=None, help="Hydra key=value overrides."),
) -> None:
    """Train the urgency classifier using the given config."""
    from omegaconf import OmegaConf

    from urgency.config import AppConfig
    from urgency.data.dataset import build_datamodule_from_config
    from urgency.models.backbones import build_model
    from urgency.training.callbacks import BestModelCallback, LeaderboardCallback, MetricsLoggerCallback
    from urgency.training.trainer import UrgencyLightningModule
    from urgency.utils.checks import check_splits_exist
    from urgency.utils.io import ensure_dir, get_git_hash, save_config
    from urgency.utils.seed import set_seed

    import lightning as pl

    cfg = _load_hydra_config(config, overrides or [])

    schema = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(schema, cfg)
    app_cfg: AppConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    set_seed(app_cfg.train.seed)

    run_dir = ensure_dir(Path(app_cfg.output_dir) / app_cfg.run_name)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "git_hash.txt").write_text(get_git_hash(), encoding="utf-8")

    check_splits_exist(Path(app_cfg.data.splits_dir))

    smoke = bool(OmegaConf.select(cfg, "_smoke_", default=False))
    dm = build_datamodule_from_config(app_cfg.data, smoke=smoke)
    dm.setup("fit")

    class_weights = dm.class_weights if app_cfg.train.class_weights else None
    model = build_model(app_cfg.model)
    pl_module = UrgencyLightningModule(model, app_cfg, class_weights)

    callbacks = [
        BestModelCallback(run_dir),
        MetricsLoggerCallback(run_dir),
        LeaderboardCallback(app_cfg, run_dir),
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="last",
            save_last=True,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=app_cfg.train.max_epochs,
        gradient_clip_val=app_cfg.train.gradient_clip_val,
        precision=app_cfg.train.precision,
        callbacks=callbacks,
        logger=pl.pytorch.loggers.TensorBoardLogger(
            save_dir=str(run_dir), name="tensorboard"
        ),
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    trainer.fit(pl_module, datamodule=dm)
    typer.echo(f"Training complete. Run saved to: {run_dir}")


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


@app.command("calibrate")
def calibrate_cmd(
    run_dir: Path = typer.Option(..., help="Path to the training run directory."),
    split: str = typer.Option("val", help="Split to use for calibration (val or train)."),
) -> None:
    """Fit temperature scaling and tune triage thresholds on a validation split."""
    import numpy as np
    import torch

    from urgency.data.dataset import build_datamodule_from_config
    from urgency.evaluation.calibration import TemperatureScaler
    from urgency.evaluation.metrics import tune_thresholds
    from urgency.models.backbones import build_model, load_model_weights
    from urgency.utils.checks import check_run_dir
    from urgency.utils.io import load_app_config, save_json

    check_run_dir(run_dir)

    app_cfg = load_app_config(run_dir / "config.yaml")

    dm = build_datamodule_from_config(app_cfg.data)
    dm.setup("fit")

    loader = dm.val_dataloader() if split == "val" else dm.train_dataloader()

    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    model = build_model(app_cfg.model)
    model = load_model_weights(model, str(ckpt_path))

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"])
            all_logits.append(logits)
            all_labels.append(batch["label"])

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)

    scaler = TemperatureScaler().fit(logits_t, labels_t)
    scaler.save(run_dir / "temperature.pt")

    if app_cfg.triage.tune_thresholds:
        import torch.nn.functional as F

        cal_logits = scaler(logits_t)
        probs = F.softmax(cal_logits, dim=1).detach().numpy()
        p_urgent = probs[:, 0]
        y_true = labels_t.numpy()

        t_high, t_low = tune_thresholds(
            p_urgent, y_true, target_sensitivity=app_cfg.triage.target_sensitivity
        )
    else:
        t_high = app_cfg.triage.t_high
        t_low = app_cfg.triage.t_low

    save_json({"t_high": t_high, "t_low": t_low}, run_dir / "thresholds.json")
    typer.echo(f"Calibration complete. T={scaler.temperature.item():.4f}, t_high={t_high:.3f}, t_low={t_low:.3f}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command("evaluate")
def evaluate_cmd(
    run_dir: Path = typer.Option(..., help="Path to the training run directory."),
    split: str = typer.Option("test", help="Split to evaluate (train, val, test)."),
) -> None:
    """Evaluate a trained model and generate all reports."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from urgency.data.dataset import build_datamodule_from_config
    from urgency.data.splits import URGENCY_CLASSES
    from urgency.evaluation.calibration import TemperatureScaler
    from urgency.evaluation.error_analysis import write_error_analysis, write_run_summary
    from urgency.evaluation.metrics import apply_triage, compute_metrics
    from urgency.evaluation.plots import save_calibration_plot, save_confusion_matrix, save_roc_curve
    from urgency.models.backbones import build_model, load_model_weights
    from urgency.utils.checks import check_run_dir
    from urgency.utils.io import ensure_dir, get_git_hash, load_app_config, load_json, save_json

    check_run_dir(run_dir)

    app_cfg = load_app_config(run_dir / "config.yaml")

    reports_dir = ensure_dir(Path(app_cfg.reports_dir))

    dm = build_datamodule_from_config(app_cfg.data)
    dm.setup("test")

    if split == "test":
        loader = dm.test_dataloader()
    elif split == "val":
        dm.setup("fit")
        loader = dm.val_dataloader()
    else:
        dm.setup("fit")
        loader = dm.train_dataloader()

    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    model = build_model(app_cfg.model)
    model = load_model_weights(model, str(ckpt_path))

    # Load calibration
    temp_path = run_dir / "temperature.pt"
    scaler = TemperatureScaler.load(temp_path) if temp_path.exists() else None

    thresh_path = run_dir / "thresholds.json"
    thresholds = load_json(thresh_path) if thresh_path.exists() else {"t_high": app_cfg.triage.t_high, "t_low": app_cfg.triage.t_low}
    t_high = float(thresholds["t_high"])
    t_low = float(thresholds["t_low"])

    all_logits, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"])
            all_logits.append(logits)
            all_labels.append(batch["label"])
            all_paths.extend(batch.get("image_path", [""] * len(batch["label"])))

    logits_t = torch.cat(all_logits)
    labels_np = torch.cat(all_labels).numpy()

    if scaler is not None:
        logits_t = scaler(logits_t)

    probs = F.softmax(logits_t, dim=1).detach().numpy()
    p_urgent = probs[:, 0]
    y_pred = np.argmax(probs, axis=1)
    triage = apply_triage(p_urgent, t_high, t_low)

    metrics = compute_metrics(
        y_true=labels_np,
        y_pred=y_pred,
        p_urgent=p_urgent,
        triage_decisions=triage,
        n_bootstrap=500,
    )

    save_json(metrics, reports_dir / "metrics.json")

    # Plots
    save_confusion_matrix(labels_np, triage, URGENCY_CLASSES, reports_dir / "confusion_matrix.png")
    y_binary = (labels_np == 0).astype(int)
    save_calibration_plot(p_urgent, y_binary, reports_dir / "calibration_plot.png")
    save_roc_curve(p_urgent, y_binary, reports_dir / "roc_curve.png")

    # Error analysis
    error_df = pd.DataFrame({
        "image_path": all_paths,
        "y_true": labels_np,
        "triage_decision": triage,
        "p_urgent": p_urgent,
    })
    write_error_analysis(error_df, reports_dir / "error_analysis.md")
    write_run_summary(app_cfg, metrics, reports_dir / "run_summary.md", get_git_hash())

    typer.echo(f"Evaluation complete. Reports saved to: {reports_dir}")
    typer.echo(f"  macro_f1:           {metrics['macro_f1']:.4f}")
    typer.echo(f"  urgent_sensitivity: {metrics['urgent_sensitivity']:.4f}")
    typer.echo(f"  roc_auc:            {metrics['roc_auc']:.4f}")
    typer.echo(f"  ece:                {metrics['ece']:.4f}")


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------


@app.command("infer")
def infer_cmd(
    run_dir: Path = typer.Option(..., help="Path to the training run directory."),
    image: Path = typer.Option(..., help="Path to the input image file."),
) -> None:
    """Run urgency inference on a single image and print the result."""
    from urgency.inference.infer import UrgencyInferer

    inferer = UrgencyInferer(run_dir)
    result = inferer.predict(image)
    typer.echo(json.dumps(result.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command("serve")
def serve_cmd(
    run_dir: Path = typer.Option(..., help="Path to the training run directory."),
    host: str = typer.Option("0.0.0.0", help="Host address."),
    port: int = typer.Option(8000, help="Port number."),
) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    from urgency.inference.api import create_app

    api = create_app(run_dir)
    uvicorn.run(api, host=host, port=port)


# ---------------------------------------------------------------------------
# check-stop
# ---------------------------------------------------------------------------


@app.command("check-stop")
def check_stop_cmd(
    leaderboard: Path = typer.Option(
        Path("reports/leaderboard.csv"), help="Path to leaderboard.csv."
    ),
    max_runs: int = typer.Option(12, help="Maximum total runs before stopping."),
    patience: int = typer.Option(3, help="Consecutive non-improving runs before stopping."),
    min_f1_delta: float = typer.Option(0.005, help="Minimum macro F1 improvement threshold."),
    min_sens_delta: float = typer.Option(0.01, help="Minimum urgent sensitivity improvement threshold."),
) -> None:
    """Check iterative improvement loop stop condition."""
    import pandas as pd

    if not leaderboard.exists():
        typer.echo("No leaderboard found. Continue training.")
        raise typer.Exit(0)

    df = pd.read_csv(leaderboard).sort_values("timestamp").reset_index(drop=True)
    total_runs = len(df)

    if total_runs == 0:
        typer.echo("No runs recorded. Continue training.")
        raise typer.Exit(0)

    if total_runs >= max_runs:
        typer.echo(f"STOP: Maximum runs reached ({total_runs}/{max_runs}).")
        raise typer.Exit(1)

    # Count consecutive non-improving runs
    best_f1 = -1.0
    best_sens = -1.0
    consecutive_no_improve = 0

    for _, row in df.iterrows():
        f1 = float(row.get("macro_f1", -1))
        sens = float(row.get("urgent_sensitivity", -1))

        if (f1 - best_f1) >= min_f1_delta or (sens - best_sens) >= min_sens_delta:
            best_f1 = max(best_f1, f1)
            best_sens = max(best_sens, sens)
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    typer.echo(f"Total runs: {total_runs}/{max_runs}")
    typer.echo(f"Consecutive non-improving: {consecutive_no_improve}/{patience}")
    typer.echo(f"Best macro_f1: {best_f1:.4f}")
    typer.echo(f"Best urgent_sensitivity: {best_sens:.4f}")

    if consecutive_no_improve >= patience:
        typer.echo(f"STOP: {patience} consecutive non-improving runs.")
        raise typer.Exit(1)

    typer.echo("Continue: improvement criterion not yet plateaued.")


# ---------------------------------------------------------------------------
# Hydra config loader helper
# ---------------------------------------------------------------------------


def _load_hydra_config(config_path: Path, overrides: list[str]):
    """Load a Hydra config from a YAML file with optional overrides.

    Uses hydra.initialize_config_dir + compose API for programmatic loading.
    This avoids Hydra taking over sys.argv.
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_path = Path(config_path).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


if __name__ == "__main__":
    app()
