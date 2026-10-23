"""Train the five Table-II models and export metrics/predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASS_IDS, CLASS_NAMES, DROPOUT, EMBEDDING_DIM, GRU_HIDDEN_DIM, OUTPUT_DIR, RANDOM_SEED
from models import MarkovPredictor, MostFrequentPredictor
from training_utils import (
    classification_metrics, fit_temperature, infer_logits, make_model,
    save_checkpoint, set_seed, softmax_probabilities, train_model,
)


def load_prepared(path: Path) -> tuple[dict, dict[int, str]]:
    data = np.load(path, allow_pickle=True)
    bundle = {key: data[key] for key in data.files}
    vocab = {str(token): int(idx) for token, idx in zip(bundle["vocab_tokens"], bundle["vocab_ids"])}
    token_id_to_text = {idx: token for token, idx in vocab.items()}
    return bundle, token_id_to_text


def split_data(bundle: dict, name: str) -> dict[str, np.ndarray]:
    return {
        key: bundle[f"{name}_{key}"]
        for key in ["X", "lengths", "y", "prefix_time", "target_time", "target_event_id", "stream_key", "prefix_event_index", "prefix_length_original", "source_file"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Table-II forecasting models.")
    parser.add_argument("--prepared", type=str, default=str(OUTPUT_DIR / "prepared_data.npz"))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    set_seed(RANDOM_SEED)
    prepared_path = Path(args.prepared).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle, token_id_to_text = load_prepared(prepared_path)
    tr, va, te = split_data(bundle, "train"), split_data(bundle, "val"), split_data(bundle, "test")
    n_classes = len(CLASS_IDS)
    vocab_size = len(token_id_to_text)
    model_config = {
        "embedding_dim": EMBEDDING_DIM, "hidden_dim": GRU_HIDDEN_DIM,
        "dropout": DROPOUT, "max_length": int(bundle["test_X"].shape[1]),
        "n_heads": 4, "n_layers": 1, "feedforward_dim": 64,
    }

    results = []
    probs_all: dict[str, np.ndarray] = {}

    # Most-frequent baseline.
    most_frequent = MostFrequentPredictor(n_classes).fit(tr["y"])
    probs = most_frequent.predict_proba(len(te["y"]))
    probs_all["prob_most_frequent"] = probs.astype(np.float32)
    results.append({"Model": "Most Frequent", **classification_metrics(te["y"], probs, n_classes)})

    # First- and second-order Markov baselines; contexts retain co-occurring
    # technique labels as event-level sets.
    for order, output_key, model_label in [
        (1, "prob_markov1", "Markov (1st order)"),
        (2, "prob_markov2", "Markov (2nd order)"),
    ]:
        model = MarkovPredictor(order=order, n_classes=n_classes, alpha=0.5)
        model.fit(tr["X"], tr["lengths"], tr["y"], token_id_to_text)
        probs = model.predict_proba(te["X"], te["lengths"], token_id_to_text)
        probs_all[output_key] = probs.astype(np.float32)
        results.append({"Model": model_label, **classification_metrics(te["y"], probs, n_classes)})

    # Neural baseline: causal Transformer.
    for model_type, output_key, model_label, checkpoint_name in [
        ("transformer", "prob_transformer", "Causal Transformer", "causal_transformer.pt"),
        ("gru_attention", "prob_gru_attention", "GRU + Attention (Proposed)", "gru_attention.pt"),
    ]:
        model = make_model(model_type, vocab_size, n_classes, model_config)
        model, _ = train_model(
            model, tr["X"], tr["lengths"], tr["y"], va["X"], va["lengths"], va["y"],
            out_dir, model_type, max_epochs=args.epochs,
        )
        val_logits, _ = infer_logits(model, va["X"], va["lengths"])
        temperature = fit_temperature(val_logits, va["y"])
        test_logits, _ = infer_logits(model, te["X"], te["lengths"])
        probs = softmax_probabilities(test_logits, temperature)
        probs_all[output_key] = probs.astype(np.float32)
        results.append({"Model": model_label, **classification_metrics(te["y"], probs, n_classes)})
        save_checkpoint(out_dir / checkpoint_name, model, model_type, vocab_size, n_classes, temperature, model_config)
        print(f"{model_label}: validation temperature={temperature:.4f}")

    order = ["Most Frequent", "Markov (1st order)", "Markov (2nd order)", "Causal Transformer", "GRU + Attention (Proposed)"]
    metrics = pd.DataFrame(results).set_index("Model").loc[order].reset_index()
    metrics.to_csv(out_dir / "table_II_prediction_results.csv", index=False, float_format="%.6f")
    print("\nTABLE II — Next-technique prediction results")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    prediction_bundle = {
        "y_test": te["y"], "class_ids": np.array(CLASS_IDS, dtype=str),
        "class_names": np.array(CLASS_NAMES, dtype=str),
        "test_prefix_time": te["prefix_time"], "test_target_time": te["target_time"],
        "test_target_event_id": te["target_event_id"], "test_stream_key": te["stream_key"],
        "test_prefix_event_index": te["prefix_event_index"], "test_prefix_length": te["lengths"],
        "test_prefix_length_original": te["prefix_length_original"],
    }
    prediction_bundle.update(probs_all)
    np.savez_compressed(out_dir / "evaluation_bundle.npz", **prediction_bundle)

    # Visualization uses separate axes for rates, avoiding a misleading single score.
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.34
    ax1.bar(x - width/2, metrics["Macro-F1"], width, label="Macro-F1")
    ax1.bar(x + width/2, metrics["Top-3 Acc."], width, label="Top-3 accuracy")
    ax1.set_ylabel("Classification metric")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace(" (Proposed)", "") for name in metrics["Model"]], rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, metrics["FPR"], marker="o", label="Macro FPR")
    ax2.set_ylabel("Macro false-positive rate")
    ax2.set_ylim(0, max(0.12, float(metrics["FPR"].max()) * 1.25))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("Current reproduction run: next-technique prediction")
    fig.tight_layout()
    fig.savefig(out_dir / "table_II_results_plot.png", dpi=180)
    plt.close(fig)

    (out_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'table_II_prediction_results.csv'}")
    print(f"Saved: {out_dir / 'evaluation_bundle.npz'}")


if __name__ == "__main__":
    main()
