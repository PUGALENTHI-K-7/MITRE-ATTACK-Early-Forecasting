"""Reproduce Table III: per-technique median early-warning lead time and coverage."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ATTENTION_CONFIDENCE_THRESHOLD, CLASS_IDS, CLASS_NAMES, LEAD_TIME_CANDIDATE_POSITIONS,
    LEAD_TIME_TARGETS_PER_CLASS, OUTPUT_DIR, RANDOM_SEED, SUSTAINED_WARNING_STEPS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sustained-warning lead time and coverage.")
    parser.add_argument("--prepared", type=str, default=str(OUTPUT_DIR / "prepared_data.npz"))
    parser.add_argument("--evaluation", type=str, default=str(OUTPUT_DIR / "evaluation_bundle.npz"))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--threshold", type=float, default=ATTENTION_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    prep = np.load(Path(args.prepared).expanduser().resolve(), allow_pickle=True)
    evaluation = np.load(Path(args.evaluation).expanduser().resolve(), allow_pickle=True)

    y = evaluation["y_test"].astype(int)
    probs = evaluation["prob_gru_attention"].astype(float)
    prefix_time = evaluation["test_prefix_time"].astype(float)
    target_time = evaluation["test_target_time"].astype(float)
    target_event_id = evaluation["test_target_event_id"].astype(str)
    stream_key = evaluation["test_stream_key"].astype(str)
    prefix_event_index = evaluation["test_prefix_event_index"].astype(int)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    # Group each future target occurrence with all preceding prefixes for which
    # it was the next distinct technique. The max sample bound matches the paper.
    rows = []
    for class_index, (class_id, class_name) in enumerate(zip(CLASS_IDS, CLASS_NAMES)):
        mask = y == class_index
        target_ids = np.unique(target_event_id[mask])
        target_order = []
        for tid in target_ids:
            idx = np.where(mask & (target_event_id == tid))[0]
            if len(idx):
                target_order.append((float(np.min(target_time[idx])), str(tid)))
        target_order.sort(key=lambda item: (item[0], item[1]))
        sampled_targets = target_order[:LEAD_TIME_TARGETS_PER_CLASS]
        lead_seconds: list[float] = []
        valid_warnings = 0

        for _, tid in sampled_targets:
            candidates = np.where(mask & (target_event_id == tid))[0]
            if not len(candidates):
                continue
            # Keep candidate positions from the same stream and order them as
            # the partial prefix grows. Select up to 20 earliest candidate rows.
            candidates = sorted(
                candidates.tolist(),
                key=lambda idx: (prefix_event_index[idx], prefix_time[idx], idx),
            )
            # Deduplicate an identical prefix index if a multi-label target made
            # repeated samples for the same technique/event.
            unique_candidates = []
            seen_pos = set()
            for idx in candidates:
                pos_key = (stream_key[idx], int(prefix_event_index[idx]))
                if pos_key not in seen_pos:
                    seen_pos.add(pos_key)
                    unique_candidates.append(idx)
            # Sample at most 20 candidate *windows* across the full period before
            # the target, rather than checking only the earliest positions (which
            # could miss a warning that becomes confident closer to the target).
            n_starts = max(0, len(unique_candidates) - SUSTAINED_WARNING_STEPS + 1)
            if n_starts <= LEAD_TIME_CANDIDATE_POSITIONS:
                candidate_starts = list(range(n_starts))
            else:
                candidate_starts = sorted(set(np.linspace(
                    0, n_starts - 1, num=LEAD_TIME_CANDIDATE_POSITIONS, dtype=int
                ).tolist()))
            valid_start = None
            for k in candidate_starts:
                window = unique_candidates[k:k + SUSTAINED_WARNING_STEPS]
                same_stream = len({stream_key[idx] for idx in window}) == 1
                consecutive = all(
                    prefix_event_index[window[j + 1]] == prefix_event_index[window[j]] + 1
                    for j in range(len(window) - 1)
                )
                sustained = len(window) == SUSTAINED_WARNING_STEPS and all(
                    pred[idx] == class_index and confidence[idx] >= args.threshold for idx in window
                )
                if same_stream and consecutive and sustained:
                    if valid_start is None or prefix_time[window[0]] < prefix_time[valid_start]:
                        valid_start = window[0]
            if valid_start is not None:
                seconds = float(target_time[valid_start] - prefix_time[valid_start])
                if np.isfinite(seconds) and seconds >= 0:
                    valid_warnings += 1
                    lead_seconds.append(seconds)

        coverage = 100.0 * valid_warnings / len(sampled_targets) if sampled_targets else 0.0
        median_lead = float(np.median(lead_seconds)) if lead_seconds else np.nan
        rows.append({
            "Technique": class_name,
            "Technique ID": class_id,
            "Sampled target instances": len(sampled_targets),
            "Valid sustained warnings": valid_warnings,
            "Median Lead (s)": median_lead,
            "Coverage (%)": coverage,
        })

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "table_III_lead_time.csv", index=False, na_rep="—", float_format="%.4f")
    pretty = result.copy()
    pretty["Median Lead (s)"] = pretty["Median Lead (s)"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    pretty["Coverage (%)"] = pretty["Coverage (%)"].map(lambda x: f"{x:.2f}")
    print("\nTABLE III — Early-warning lead time per technique")
    print(pretty[["Technique", "Median Lead (s)", "Coverage (%)"]].to_string(index=False))
    print(f"Saved: {out_dir / 'table_III_lead_time.csv'}")


if __name__ == "__main__":
    main()
