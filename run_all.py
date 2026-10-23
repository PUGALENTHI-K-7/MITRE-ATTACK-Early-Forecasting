"""One-command workflow for data preparation and all result tables."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import OUTPUT_DIR
from data_pipeline import prepare_bundle, resolve_data_dir


def run(script: str, prepared: Path, output_dir: Path, epochs: int) -> None:
    command = [
        sys.executable, str(Path(__file__).resolve().parent / script),
        "--prepared", str(prepared), "--output-dir", str(output_dir),
    ]
    if script in {"lead_time_analysis.py", "faithfulness_analysis.py", "ablation_analysis.py"}:
        command += ["--evaluation", str(output_dir / "evaluation_bundle.npz")]
    if script == "faithfulness_analysis.py":
        command += ["--checkpoint", str(output_dir / "gru_attention.pt")]
    if script in {"train_evaluate.py", "ablation_analysis.py"}:
        command += ["--epochs", str(epochs)]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper's full forecasting evaluation pipeline.")
    parser.add_argument("--data-dir", type=str, default=None, help="Extracted dataset folder containing the 16 period CSVs.")
    parser.add_argument("--dataset-zip", type=str, default=None, help="Dataset ZIP path (extracted automatically if --data-dir is omitted).")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Directory for tables, plots, checkpoints and diagnostic files.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum neural model epochs (paper setting: 15).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(args.data_dir, args.dataset_zip)
    print(f"Dataset folder: {data_dir}")
    print(f"Output folder:  {output_dir}")
    prepare_bundle(data_dir, output_dir)

    prepared = output_dir / "prepared_data.npz"
    run("train_evaluate.py", prepared, output_dir, args.epochs)
    run("lead_time_analysis.py", prepared, output_dir, args.epochs)
    run("faithfulness_analysis.py", prepared, output_dir, args.epochs)
    run("ablation_analysis.py", prepared, output_dir, args.epochs)
    print("\nAll stages completed. Inspect CSV tables and preprocessing_diagnostics.json before using values in the paper.")


if __name__ == "__main__":
    main()
