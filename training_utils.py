"""Shared training, inference, calibration and metric functions."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, DROPOUT, EARLY_STOPPING_PATIENCE, EMBEDDING_DIM, GRU_HIDDEN_DIM, LEARNING_RATE, MAX_EPOCHS, RANDOM_SEED
from models import CausalTransformerModel, GRUAttentionModel, GRUWithoutAttentionModel


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The paper used CPU; keep deterministic CPU behavior as the default.
    torch.use_deterministic_algorithms(False)


def make_model(model_type: str, vocab_size: int, n_classes: int, config: dict | None = None) -> nn.Module:
    cfg = dict(config or {})
    common = {
        "vocab_size": int(vocab_size), "n_classes": int(n_classes),
        "embedding_dim": int(cfg.get("embedding_dim", EMBEDDING_DIM)),
        "dropout": float(cfg.get("dropout", DROPOUT)),
    }
    if model_type in {"gru_attention", "gru_no_attention"}:
        common["hidden_dim"] = int(cfg.get("hidden_dim", GRU_HIDDEN_DIM))
        cls = GRUAttentionModel if model_type == "gru_attention" else GRUWithoutAttentionModel
        return cls(**common)
    if model_type == "transformer":
        common.update({
            "max_length": int(cfg.get("max_length", 15)),
            "n_heads": int(cfg.get("n_heads", 4)),
            "n_layers": int(cfg.get("n_layers", 1)),
            "feedforward_dim": int(cfg.get("feedforward_dim", 64)),
        })
        return CausalTransformerModel(**common)
    raise ValueError(f"Unknown model type: {model_type}")


def batch_tensors(split: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(split["X"], dtype=torch.long),
        torch.as_tensor(split["lengths"], dtype=torch.long),
        torch.as_tensor(split["y"], dtype=torch.long),
    )


def train_model(model: nn.Module, X_train: np.ndarray, lengths_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, lengths_val: np.ndarray, y_val: np.ndarray,
                output_dir: Path, model_name: str, max_epochs: int = MAX_EPOCHS) -> tuple[nn.Module, list[dict]]:
    set_seed(RANDOM_SEED)
    device = torch.device("cpu")
    model.to(device)
    xtr = torch.as_tensor(X_train, dtype=torch.long)
    ltr = torch.as_tensor(lengths_train, dtype=torch.long)
    ytr = torch.as_tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(xtr, ltr, ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    xval = torch.as_tensor(X_val, dtype=torch.long)
    lval = torch.as_tensor(lengths_val, dtype=torch.long)
    yval = torch.as_tensor(y_val, dtype=torch.long)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        for xb, lb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), lb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            train_seen += len(yb)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_seen = 0
            for start in range(0, len(xval), BATCH_SIZE):
                xb = xval[start:start+BATCH_SIZE]
                lb = lval[start:start+BATCH_SIZE]
                yb = yval[start:start+BATCH_SIZE]
                logits = model(xb, lb)
                loss = criterion(logits, yb)
                val_loss_sum += float(loss.item()) * len(yb)
                val_seen += len(yb)
        train_loss = train_loss_sum / max(train_seen, 1)
        val_loss = val_loss_sum / max(val_seen, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{model_name}] epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(history).to_csv(output_dir / f"training_history_{model_name}.csv", index=False)
    print(f"[{model_name}] selected epoch={best_epoch}; best validation loss={best_val_loss:.4f}")
    return model, history


def infer_logits(model: nn.Module, X: np.ndarray, lengths: np.ndarray, batch_size: int = BATCH_SIZE,
                 return_attention: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    device = torch.device("cpu")
    x = torch.as_tensor(X, dtype=torch.long)
    lens = torch.as_tensor(lengths, dtype=torch.long)
    all_logits = []
    all_attention = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start+batch_size].to(device)
            lb = lens[start:start+batch_size].to(device)
            if return_attention:
                logits, attention = model(xb, lb, return_attention=True)
                if attention is not None:
                    all_attention.append(attention.cpu().numpy())
            else:
                logits = model(xb, lb)
            all_logits.append(logits.cpu().numpy())
    logits_array = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0), dtype=float)
    attention_array = np.concatenate(all_attention, axis=0) if all_attention else None
    return logits_array, attention_array


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a scalar temperature by minimizing validation negative log likelihood."""
    if len(logits) == 0:
        return 1.0
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    y_t = torch.as_tensor(y_true, dtype=torch.long)
    log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temp).clamp(0.05, 20.0)
        loss = criterion(logits_t / temperature, y_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        return float(torch.exp(log_temp).clamp(0.05, 20.0).detach().item())
    except Exception:
        return 1.0


def softmax_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if len(logits) == 0:
        return np.empty_like(logits, dtype=float)
    z = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.maximum(ez.sum(axis=1, keepdims=True), 1e-12)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> dict[str, float]:
    if len(y_true) == 0:
        return {"Macro-F1": 0.0, "Top-3 Acc.": 0.0, "FPR": 0.0}
    y_pred = probabilities.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0)
    k = min(3, n_classes)
    topk = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    top3 = np.any(topk == np.asarray(y_true)[:, None], axis=1).mean()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(float)
    fprs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fprs.append(fp / (fp + tn) if fp + tn > 0 else 0.0)
    return {"Macro-F1": float(macro_f1), "Top-3 Acc.": float(top3), "FPR": float(np.mean(fprs))}


def save_checkpoint(path: Path, model: nn.Module, model_type: str, vocab_size: int,
                    n_classes: int, temperature: float, config: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "model_type": model_type,
        "vocab_size": int(vocab_size), "n_classes": int(n_classes),
        "temperature": float(temperature), "config": dict(config or {}),
    }, path)
