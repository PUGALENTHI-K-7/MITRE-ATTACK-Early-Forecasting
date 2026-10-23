"""Reproduce Table V: GRU attention ablation and prefix-length sensitivity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLASS_IDS, DROPOUT, EMBEDDING_DIM, GRU_HIDDEN_DIM, OUTPUT_DIR
from training_utils import (
    classification_metrics, fit_temperature, infer_logits, make_model,
    save_checkpoint, softmax_probabilities, train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train no-attention GRU and create ablation table.")
    parser.add_argument("--prepared", type=str, default=str(OUTPUT_DIR / "prepared_data.npz"))
    parser.add_argument("--evaluation", type=str, default=str(OUTPUT_DIR / "evaluation_bundle.npz"))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    data = np.load(Path(args.prepared).expanduser().resolve(), allow_pickle=True)
    ev = np.load(Path(args.evaluation).expanduser().resolve(), allow_pickle=True)
    Xtr, Ltr, ytr = data["train_X"], data["train_lengths"], data["train_y"]
    Xva, Lva, yva = data["val_X"], data["val_lengths"], data["val_y"]
    Xte, Lte, yte = data["test_X"], data["test_lengths"], data["test_y"]
    vocab_size = len(data["vocab_tokens"])
    n_classes = len(CLASS_IDS)
    cfg = {"embedding_dim": EMBEDDING_DIM, "hidden_dim": GRU_HIDDEN_DIM, "dropout": DROPOUT, "max_length": int(Xte.shape[1])}

    no_attention = make_model("gru_no_attention", vocab_size, n_classes, cfg)
    no_attention, _ = train_model(no_attention, Xtr, Ltr, ytr, Xva, Lva, yva, out_dir, "gru_no_attention", max_epochs=args.epochs)
    val_logits, _ = infer_logits(no_attention, Xva, Lva)
    temperature = fit_temperature(val_logits, yva)
    test_logits, _ = infer_logits(no_attention, Xte, Lte)
    probs_no_attention = softmax_probabilities(test_logits, temperature)
    save_checkpoint(out_dir / "gru_no_attention.pt", no_attention, "gru_no_attention", vocab_size, n_classes, temperature, cfg)

    no_attn_metrics = classification_metrics(yte, probs_no_attention, n_classes)
    attn_probs = ev["prob_gru_attention"].astype(float)
    attn_metrics = classification_metrics(yte, attn_probs, n_classes)
    rows = [
        {"Configuration": "GRU without attention", "Macro-F1": no_attn_metrics["Macro-F1"], "Top-3 Acc.": no_attn_metrics["Top-3 Acc."]},
        {"Configuration": "GRU with attention (proposed)", "Macro-F1": attn_metrics["Macro-F1"], "Top-3 Acc.": attn_metrics["Top-3 Acc."]},
    ]
    prefix_lengths = ev["test_prefix_length"].astype(int)
    for k, label in [(3, "Short prefix (k ≤ 3)"), (6, "Medium prefix (k ≤ 6)"), (10, "Long prefix (k ≤ 10)")]:
        mask = prefix_lengths <= k
        metrics = classification_metrics(yte[mask], attn_probs[mask], n_classes) if mask.any() else {"Macro-F1": 0.0, "Top-3 Acc.": 0.0}
        rows.append({"Configuration": label, "Macro-F1": metrics["Macro-F1"], "Top-3 Acc.": metrics["Top-3 Acc."]})

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "table_V_ablation_prefix_sensitivity.csv", index=False, float_format="%.6f")
    print("\nTABLE V — Ablation and prefix-length sensitivity")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {out_dir / 'table_V_ablation_prefix_sensitivity.csv'}")


if __name__ == "__main__":
    main()
