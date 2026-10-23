"""Load Windows-APT period CSVs and build causally ordered prefix/target examples.

The dataset's schemas differ by collection period; this module normalizes the
most relevant fields without assuming that one CSV schema applies to all files.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import (
    AGENT_ID_COLUMNS, AGENT_NAME_COLUMNS, CLASS_IDS, COMMAND_COLUMNS,
    DEFAULT_DATA_DIR, DEFAULT_DATASET_ZIP, ID_TO_NAME, MAX_PREFIX_LENGTH,
    PROCESS_ID_COLUMNS, PROCESS_IMAGE_COLUMNS, RANDOM_SEED, TECHNIQUE_ALIASES,
    TECHNIQUE_ID_COLUMNS, TECHNIQUE_ID_FALLBACK_COLUMNS, TECHNIQUE_NAME_COLUMNS,
    TEST_FILES, TIMESTAMP_COLUMNS, TRAIN_FILES,
)

# Only read these columns, while allowing heterogeneous CSV schemas.
NEEDED_COLUMNS = set(
    TIMESTAMP_COLUMNS + AGENT_NAME_COLUMNS + AGENT_ID_COLUMNS
    + TECHNIQUE_NAME_COLUMNS + TECHNIQUE_ID_COLUMNS + TECHNIQUE_ID_FALLBACK_COLUMNS
    + PROCESS_IMAGE_COLUMNS + PROCESS_ID_COLUMNS + COMMAND_COLUMNS
)


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(columns)
    return next((col for col in candidates if col in colset), None)


def parse_list_value(value: Any) -> list[str]:
    """Parse JSON-ish lists, comma-delimited ID fields, or a single string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text in {" ", "-"}:
        return []
    # JSON lists are common in _source.rule.* fields.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (ValueError, SyntaxError):
                pass
    # Older schema: T1059,T1059.001
    if re.fullmatch(r"\s*T\d{4}(?:\.\d{3})?(?:\s*,\s*T\d{4}(?:\.\d{3})?)*\s*", text, re.I):
        return [token.strip().upper() for token in text.split(",") if token.strip()]
    return [text]


def canonical_technique(token: str) -> str:
    token = str(token).strip().strip('"\'')
    if not token:
        return ""
    match = re.fullmatch(r"(T\d{4}(?:\.\d{3})?)", token, flags=re.I)
    if match:
        return match.group(1).upper()
    compact = re.sub(r"\s+", " ", token).strip().casefold()
    alias = TECHNIQUE_ALIASES.get(compact)
    if alias:
        return alias
    # Preserve non-selected technique names as tokens; they may appear in the
    # input history, although the prediction target uses only the 12 classes.
    return "NAME:" + re.sub(r"\s+", " ", token).strip().casefold()


def parse_row_techniques(row: dict[str, Any], columns: list[str]) -> tuple[str, ...]:
    """Return an order-independent technique set for a source record.

    When IDs are present, they take precedence over technique-name strings. This
    prevents one ATT&CK technique from being represented twice as both e.g. an
    ID token and a separate name token.
    """
    id_cols = [c for c in TECHNIQUE_ID_COLUMNS if c in columns]
    # Deliberately exclude _source.data.sca.check.compliance.mitre_techniques:
    # it is a compliance/check field, not an event-level ATT&CK rule annotation.
    name_cols = [c for c in TECHNIQUE_NAME_COLUMNS if c in columns]
    id_tokens: set[str] = set()
    name_tokens: set[str] = set()

    for col in id_cols:
        for item in parse_list_value(row.get(col, "")):
            canonical = canonical_technique(item)
            if canonical:
                id_tokens.add(canonical)
    for col in name_cols:
        for item in parse_list_value(row.get(col, "")):
            canonical = canonical_technique(item)
            if canonical:
                name_tokens.add(canonical)

    if id_tokens:
        # Preserve name-only selected IDs if available, but otherwise do not
        # double-count unselected ID/name pairs as separate event techniques.
        selected_names = {token for token in name_tokens if token in CLASS_IDS}
        id_tokens.update(selected_names)
        return tuple(sorted(id_tokens))
    return tuple(sorted(name_tokens))


def parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    text = text.replace(" @ ", " ")
    return pd.to_datetime(text, errors="coerce", utc=False)


def extract_dataset_zip(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # Avoid extracting the archive outside the expected data directory.
        zf.extractall(destination)
    roots = list(destination.rglob("scenario_manifest.csv"))
    if roots:
        return roots[0].parent
    # Some exports may omit supplementary files; locate the period CSVs instead.
    for path in destination.rglob("28-October-01-November.csv"):
        return path.parent
    raise FileNotFoundError(f"Could not find dataset CSVs after extracting {zip_path}")


def resolve_data_dir(data_dir: str | Path | None = None, dataset_zip: str | Path | None = None) -> Path:
    if data_dir:
        candidate = Path(data_dir).expanduser().resolve()
        if candidate.exists() and any((candidate / f).exists() for f in TRAIN_FILES):
            return candidate
        # It may be the parent folder containing the archive's own named root.
        if candidate.exists():
            for file in candidate.rglob(TRAIN_FILES[0]):
                return file.parent
        raise FileNotFoundError(f"Cannot find expected per-period CSVs under --data-dir {candidate}")

    if DEFAULT_DATA_DIR.exists() and any((DEFAULT_DATA_DIR / f).exists() for f in TRAIN_FILES):
        return DEFAULT_DATA_DIR

    zip_path = Path(dataset_zip).expanduser().resolve() if dataset_zip else DEFAULT_DATASET_ZIP
    if zip_path.exists():
        return extract_dataset_zip(zip_path, DEFAULT_DATA_DIR.parent)

    raise FileNotFoundError(
        "Dataset not found. Pass --dataset-zip 'Windows-APT 2025 A Dataset for APT-Inspired Attack.zip' "
        "or --data-dir pointing at the extracted dataset folder."
    )


def load_period_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=lambda col: col in NEEDED_COLUMNS,
        low_memory=False,
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )
    raw_row_count = int(len(frame))
    cols = list(frame.columns)
    ts_col = first_existing(cols, TIMESTAMP_COLUMNS)
    agent_name_col = first_existing(cols, AGENT_NAME_COLUMNS)
    agent_id_col = first_existing(cols, AGENT_ID_COLUMNS)
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {path.name}. Columns found: {cols}")

    normalized = pd.DataFrame(index=frame.index)
    # Parse timestamps vectorially; the archive has >100k rows, so per-row
    # pd.to_datetime calls would be needlessly slow.
    normalized["timestamp"] = pd.to_datetime(
        frame[ts_col].fillna("").astype(str).str.replace(" @ ", " ", regex=False),
        errors="coerce",
    )
    normalized["agent_name"] = frame[agent_name_col].fillna("").astype(str).str.strip() if agent_name_col else "unknown-agent"
    normalized["agent_id"] = frame[agent_id_col].fillna("").astype(str).str.strip() if agent_id_col else "unknown-id"
    normalized["source_file"] = path.name
    normalized["techniques"] = [parse_row_techniques(row, cols) for row in frame.to_dict(orient="records")]

    image_col = first_existing(cols, PROCESS_IMAGE_COLUMNS)
    pid_col = first_existing(cols, PROCESS_ID_COLUMNS)
    command_col = first_existing(cols, COMMAND_COLUMNS)
    normalized["process_image"] = frame[image_col].fillna("").astype(str).str.strip() if image_col else ""
    normalized["process_id"] = frame[pid_col].fillna("").astype(str).str.strip() if pid_col else ""
    normalized["command_line"] = frame[command_col].fillna("").astype(str).str.strip() if command_col else ""

    normalized = normalized.dropna(subset=["timestamp"])
    # The paper's main sequence is built from ATT&CK-annotated records only.
    normalized = normalized[normalized["techniques"].map(bool)].copy()
    # The paper reports 10 training and 29 test streams. Agent ID is the
    # stable key in these records; names can change across export versions.
    agent_key = normalized["agent_id"].where(normalized["agent_id"].astype(str).str.strip().ne(""), normalized["agent_name"])
    normalized["stream_key"] = normalized["source_file"].astype(str) + "::agent::" + agent_key.astype(str)
    normalized.attrs["raw_rows"] = raw_row_count
    return normalized


def compress_repeated_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Compress repeated signatures within one second, within each agent/file stream."""
    if frame.empty:
        return frame
    frame = frame.sort_values(["stream_key", "timestamp"], kind="mergesort").copy()
    kept_rows = []
    last_by_signature: dict[tuple, pd.Timestamp] = {}
    for row in frame.itertuples(index=False, name=None):
        record = dict(zip(frame.columns, row))
        sig = (
            record["stream_key"], tuple(record["techniques"]),
            str(record["process_image"]), str(record["process_id"]), str(record["command_line"]),
        )
        ts = record["timestamp"]
        previous = last_by_signature.get(sig)
        if previous is not None and (ts - previous).total_seconds() <= 1.0 and (ts - previous).total_seconds() >= 0:
            # Refresh the last-observed duplicate time so a dense burst remains
            # compressed rather than emitting every other repeated event.
            last_by_signature[sig] = ts
            continue
        last_by_signature[sig] = ts
        kept_rows.append(record)
    if not kept_rows:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(kept_rows, columns=frame.columns)


def load_streams(data_dir: Path) -> tuple[dict, dict, dict]:
    """Load and compress training and test streams; return stream map and diagnostics."""
    from config import TRAIN_FILES, TEST_FILES

    streams: dict[str, list[dict]] = {}
    diagnostics = {"files": [], "raw_rows": 0, "annotated_rows": 0, "compressed_events": 0}
    all_periods = [(filename, "train") for filename in TRAIN_FILES] + [(filename, "test") for filename in TEST_FILES]
    split_of_stream: dict[str, str] = {}
    annotated_frames = []

    for filename, split in all_periods:
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected period file missing: {file_path}")
        frame = load_period_file(file_path)
        raw_rows = int(frame.attrs.get("raw_rows", len(frame)))
        diagnostics["raw_rows"] += raw_rows
        diagnostics["annotated_rows"] += len(frame)
        diagnostics["files"].append({"file": filename, "split": split, "raw_rows": raw_rows, "annotated_rows": int(len(frame))})
        annotated_frames.append(frame)
        for key in frame["stream_key"].unique():
            split_of_stream[key] = split

    if not annotated_frames:
        raise ValueError("No files loaded. Check that the expected period CSVs exist.")
    merged = pd.concat(annotated_frames, ignore_index=True)
    compressed = compress_repeated_events(merged)
    diagnostics["compressed_events"] = int(len(compressed))

    # Do not join distinct source period files into one continuous stream. This is
    # deliberate because the paper's protocol treats each period file separately.
    for stream_key, group in compressed.groupby("stream_key", sort=False):
        group = group.sort_values("timestamp", kind="mergesort")
        events = []
        for event_index, row in enumerate(group.itertuples(index=False)):
            events.append({
                "timestamp": row.timestamp,
                "techniques": tuple(row.techniques),
                "stream_key": stream_key,
                "event_index": event_index,
                "source_file": row.source_file,
            })
        streams[stream_key] = events

    return streams, split_of_stream, diagnostics


def selected_targets(techniques: Iterable[str]) -> list[str]:
    return sorted(set(techniques).intersection(CLASS_IDS), key=CLASS_IDS.index)


def build_examples(streams: dict, split_of_stream: dict, split: str, max_prefix: int = MAX_PREFIX_LENGTH) -> list[dict]:
    """Create prefix/target samples at the next technique-set transition.

    Adjacent repeated observations with the same technique set are treated as
    repetition of the currently observed technique. A target is the first
    subsequent event whose technique set differs. If that immediate transition
    does not contain any of the 12 target classes not already present in the
    current set, no sample is created; the code does not skip over an unmodeled
    technique and incorrectly call a later class the "next" technique.

    If the next event contains multiple new target classes, each target is
    represented independently using the same prefix; CSV list order is ignored.
    """
    examples = []
    for stream_key, events in streams.items():
        if split_of_stream.get(stream_key) != split or len(events) < 2:
            continue
        n_events = len(events)
        # Find consecutive runs of identical technique sets. All prefixes that
        # end in one run point to the first event of the immediately following run.
        run_bounds = []
        run_start = 0
        for pos in range(1, n_events):
            if tuple(events[pos]["techniques"]) != tuple(events[pos - 1]["techniques"]):
                run_bounds.append((run_start, pos))
                run_start = pos
        run_bounds.append((run_start, n_events))

        for run_i, (start, end) in enumerate(run_bounds[:-1]):
            target_pos = run_bounds[run_i + 1][0]
            observed_set = set(events[start]["techniques"])
            target_event = events[target_pos]
            target_labels = [label for label in selected_targets(target_event["techniques"]) if label not in observed_set]
            if not target_labels:
                continue
            # The paper reports standard single-label Macro-F1/Top-3 metrics,
            # while the logs can occasionally have co-occurring target labels.
            # Keep the input event as a set, but resolve a multi-label target
            # deterministically using the Table-I class order so each prefix
            # contributes one multiclass sample instead of duplicating samples.
            target_label = target_labels[0]
            for i in range(start, end):
                event = events[i]
                prefix = events[max(0, i - max_prefix + 1):i + 1]
                examples.append({
                    "prefix": [e["techniques"] for e in prefix],
                    "prefix_length": len(prefix),
                    "y_label": target_label,
                    "prefix_time": event["timestamp"],
                    "target_time": target_event["timestamp"],
                    "target_event_id": f"{stream_key}::{target_event['event_index']}",
                    "target_event_index": int(target_event["event_index"]),
                    "prefix_event_index": int(event["event_index"]),
                    "stream_key": stream_key,
                    "source_file": event["source_file"],
                })
    examples.sort(key=lambda ex: (ex["prefix_time"], ex["stream_key"], ex["prefix_event_index"], CLASS_IDS.index(ex["y_label"])) )
    return examples


def build_vocab(streams: dict, split_of_stream: dict) -> tuple[dict[str, int], dict[int, str]]:
    """Fit an input technique vocabulary from training-period streams only."""
    tokens = set(CLASS_IDS)
    for stream_key, events in streams.items():
        if split_of_stream.get(stream_key) == "train":
            for event in events:
                tokens.update(event["techniques"])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in sorted(tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    inv_vocab = {index: token for token, index in vocab.items()}
    return vocab, inv_vocab


def encode_examples(examples: list[dict], vocab: dict[str, int], max_prefix: int = MAX_PREFIX_LENGTH) -> dict[str, np.ndarray]:
    max_labels_per_event = max(
        1,
        min(12, max((len(ex_event) for ex in examples for ex_event in ex["prefix"]), default=1)),
    )
    n = len(examples)
    X = np.zeros((n, max_prefix, max_labels_per_event), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int16)
    y = np.zeros(n, dtype=np.int64)
    prefix_time = np.zeros(n, dtype=np.float64)
    target_time = np.zeros(n, dtype=np.float64)
    target_event_id = np.empty(n, dtype=object)
    stream_key = np.empty(n, dtype=object)
    prefix_event_index = np.zeros(n, dtype=np.int32)
    prefix_length_original = np.zeros(n, dtype=np.int16)
    source_file = np.empty(n, dtype=object)

    from config import CLASS_TO_INDEX
    for row_idx, ex in enumerate(examples):
        eventsets = ex["prefix"][-max_prefix:]
        lengths[row_idx] = len(eventsets)
        prefix_length_original[row_idx] = ex["prefix_length"]
        for pos, techs in enumerate(eventsets):
            ids = [vocab.get(token, vocab["<UNK>"]) for token in techs]
            ids = list(dict.fromkeys(ids))[:max_labels_per_event]
            if ids:
                X[row_idx, pos, :len(ids)] = ids
        y[row_idx] = CLASS_TO_INDEX[ex["y_label"]]
        prefix_time[row_idx] = pd.Timestamp(ex["prefix_time"]).timestamp()
        target_time[row_idx] = pd.Timestamp(ex["target_time"]).timestamp()
        target_event_id[row_idx] = ex["target_event_id"]
        stream_key[row_idx] = ex["stream_key"]
        prefix_event_index[row_idx] = ex["prefix_event_index"]
        source_file[row_idx] = ex["source_file"]

    return {
        "X": X, "lengths": lengths, "y": y,
        "prefix_time": prefix_time, "target_time": target_time,
        "target_event_id": target_event_id, "stream_key": stream_key,
        "prefix_event_index": prefix_event_index,
        "prefix_length_original": prefix_length_original,
        "source_file": source_file,
    }


def prepare_bundle(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    streams, split_of_stream, diagnostics = load_streams(data_dir)
    vocab, inv_vocab = build_vocab(streams, split_of_stream)

    examples_by_split = {
        "train_pool": build_examples(streams, split_of_stream, "train"),
        "test": build_examples(streams, split_of_stream, "test"),
    }
    train_pool = examples_by_split["train_pool"]
    n_train = int(len(train_pool) * 0.8)
    train_examples = train_pool[:n_train]
    val_examples = train_pool[n_train:]
    if not train_examples or not val_examples or not examples_by_split["test"]:
        raise RuntimeError(
            f"Insufficient examples created: train={len(train_examples)}, val={len(val_examples)}, "
            f"test={len(examples_by_split['test'])}. Review dataset parsing and annotation columns."
        )

    arrays = {}
    for split_name, examples in [("train", train_examples), ("val", val_examples), ("test", examples_by_split["test"])]:
        encoded = encode_examples(examples, vocab)
        for key, value in encoded.items():
            arrays[f"{split_name}_{key}"] = value

    arrays["vocab_tokens"] = np.array(list(vocab.keys()), dtype=str)
    arrays["vocab_ids"] = np.array(list(vocab.values()), dtype=np.int32)
    arrays["class_ids"] = np.array(CLASS_IDS, dtype=str)
    arrays["class_names"] = np.array([ID_TO_NAME[x] for x in CLASS_IDS], dtype=str)
    np.savez_compressed(output_dir / "prepared_data.npz", **arrays)
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    # Store the preprocessing diagnostics and a human-readable manifest.
    diagnostics.update({
        "data_dir": str(data_dir),
        "stream_count": int(len(streams)),
        "train_stream_count": int(sum(1 for k in streams if split_of_stream.get(k) == "train")),
        "test_stream_count": int(sum(1 for k in streams if split_of_stream.get(k) == "test")),
        "vocab_size_including_special_tokens": int(len(vocab)),
        "train_prefix_target_pairs": int(len(train_examples)),
        "validation_prefix_target_pairs": int(len(val_examples)),
        "test_prefix_target_pairs": int(len(examples_by_split["test"])),
        "max_prefix_length": MAX_PREFIX_LENGTH,
        "seed": RANDOM_SEED,
    })
    (output_dir / "preprocessing_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, default=str))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Windows-APT data for next-technique forecasting.")
    parser.add_argument("--data-dir", type=str, default=None, help="Extracted dataset folder containing period CSVs.")
    parser.add_argument("--dataset-zip", type=str, default=None, help="Path to the supplied dataset ZIP; extracted automatically if needed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for prepared_data.npz and diagnostics.")
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir, args.dataset_zip)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (Path(__file__).resolve().parent / "outputs")
    prepare_bundle(data_dir, output_dir)


if __name__ == "__main__":
    main()
