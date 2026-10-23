"""Reproduce Table IV: masking-based attention faithfulness analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CLASS_IDS, FAITHFULNESS_SAMPLE_SIZE, OUTPUT_DIR, RANDOM_SEED
from training_utils import make_model, softmax_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attention-ranked masking tests on correct predictions.")
    parser.add_argument("--prepared", type=str, default=str(OUTPUT_DIR / "prepared_data.npz"))
    parser.add_argument("--evaluation", type=str, default=str(OUTPUT_DIR / "evaluation_bundle.npz"))
    parser.add_argument("--checkpoint", type=str, default=str(OUTPUT_DIR / "gru_attention.pt"))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--sample-size", type=int, default=FAITHFULNESS_SAMPLE_SIZE)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    data = np.load(Path(args.prepared).expanduser().resolve(), allow_pickle=True)
    ev = np.load(Path(args.evaluation).expanduser().resolve(), allow_pickle=True)
    checkpoint = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu", weights_only=False)

    vocab_size = int(checkpoint["vocab_size"])
    n_classes = int(checkpoint["n_classes"])
    model_type = checkpoint["model_type"]
    if model_type != "gru_attention":
        raise ValueError(f"Expected gru_attention checkpoint, got {model_type}")
    model = make_model(model_type, vocab_size, n_classes, checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    temperature = float(checkpoint.get("temperature", 1.0))

    X = data["test_X"].astype(np.int64)
    lengths = data["test_lengths"].astype(np.int64)
    y_true = ev["y_test"].astype(int)
    baseline_prob = ev["prob_gru_attention"].astype(float)
    pred = baseline_prob.argmax(axis=1)
    correct_idx = np.where(pred == y_true)[0][:max(0, int(args.sample_size))]
    if not len(correct_idx):
        raise RuntimeError("No correctly classified test examples are available for faithfulness analysis.")

    xb = torch.as_tensor(X[correct_idx], dtype=torch.long)
    lb = torch.as_tensor(lengths[correct_idx], dtype=torch.long)
    with torch.no_grad():
        logits, attention = model(xb, lb, return_attention=True)
    attention = attention.cpu().numpy()

    base_values = []
    top1_values = []
    top2_values = []
    random_values = []
    rng = np.random.default_rng(RANDOM_SEED)

    for local_i, global_i in enumerate(correct_idx):
        original_class = int(pred[global_i])
        original_probability = float(baseline_prob[global_i, original_class])
        base_values.append(original_probability)
        length = int(lengths[global_i])
        valid_weights = attention[local_i, :length]
        positions = np.argsort(valid_weights)[::-1]
        top1 = int(positions[0])
        top2 = positions[:min(2, length)].tolist()
        random_pos = int(rng.integers(0, max(length, 1)))

        for positions_to_mask, sink in [([top1], top1_values), (top2, top2_values), ([random_pos], random_values)]:
            masked = X[global_i:global_i+1].copy()
            for p in positions_to_mask:
                masked[0, p, :] = 0
            with torch.no_grad():
                masked_logits = model(torch.as_tensor(masked, dtype=torch.long), torch.as_tensor([length], dtype=torch.long))
            masked_probs = softmax_probabilities(masked_logits.cpu().numpy(), temperature)[0]
            sink.append(float(masked_probs[original_class]))

    baseline_mean = float(np.mean(base_values))
    conditions = [
        ("Unmasked (baseline)", baseline_mean),
        ("Top-1 position masked", float(np.mean(top1_values))),
        ("Top-2 positions masked", float(np.mean(top2_values))),
        ("Random position masked", float(np.mean(random_values))),
    ]
    rows = [{"Condition": label, "Avg. Prob.": value, "Prob. Drop": baseline_mean - value} for label, value in conditions]
    # Ensure reference baseline is exactly zero drop to avoid floating point display noise.
    rows[0]["Prob. Drop"] = 0.0
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "table_IV_faithfulness.csv", index=False, float_format="%.6f")
    print(f"\nTABLE IV — Attention masking faithfulness (n={len(correct_idx)})")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {out_dir / 'table_IV_faithfulness.csv'}")


if __name__ == "__main__":
    main()
