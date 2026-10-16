"""Recompute paired metrics from the saved experiment arrays."""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize(root=PROJECT_ROOT):
    """Return one record per experiment; fail on incomplete or invalid artifacts."""
    records = []
    for folder in sorted((root / "results").glob("*/*")):
        if not folder.is_dir() or folder.parent.name == "exploration":
            continue
        targets = np.load(folder / "targets.npy").astype(np.float64).reshape(-1)
        predictions = np.load(folder / "predictions.npy").astype(np.float64).reshape(-1)
        losses = np.load(folder / "losses.npy")
        if (
            targets.shape != predictions.shape
            or not np.isfinite(
                np.concatenate([targets, predictions, losses.reshape(-1)])
            ).all()
        ):
            raise ValueError(f"Invalid saved arrays: {folder}")
        errors = predictions - targets
        records.append(
            {
                "model": folder.parent.name,
                "variant": folder.name,
                "path": folder.relative_to(root).as_posix(),
                "n_test": len(targets),
                "epochs": int(losses.size),
                "mae_dex": float(np.abs(errors).mean()),
                "rmse_dex": float(np.sqrt(np.square(errors).mean())),
            }
        )
    return records


if __name__ == "__main__":
    records = summarize()
    destination = PROJECT_ROOT / "results/metrics.json"
    destination.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(
            f'{record["model"]:12s} {record["variant"]:22s} '
            f'MAE={record["mae_dex"]:.4f} RMSE={record["rmse_dex"]:.4f} '
            f'n={record["n_test"]}'
        )
