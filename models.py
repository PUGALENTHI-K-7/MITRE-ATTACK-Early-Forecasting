"""Baseline and neural architectures for ATT&CK next-technique forecasting."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import torch
from torch import nn

from config import CLASS_IDS


class MostFrequentPredictor:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.probabilities_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "MostFrequentPredictor":
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=self.n_classes).astype(float)
        self.probabilities_ = counts / max(counts.sum(), 1.0)
        return self

    def predict_proba(self, n_rows: int) -> np.ndarray:
        if self.probabilities_ is None:
            raise RuntimeError("Call fit first.")
        return np.repeat(self.probabilities_[None, :], n_rows, axis=0)


class MarkovPredictor:
    """Smoothed transition model conditioned on one or two recent event-label sets.

    A context is a tuple of sorted label-index tuples; this preserves co-occurring
    labels as sets instead of inventing an order inside a multi-technique event.
    """
    def __init__(self, order: int, n_classes: int, alpha: float = 0.5):
        if order not in (1, 2):
            raise ValueError("Markov order must be 1 or 2")
        self.order = order
        self.n_classes = n_classes
        self.alpha = float(alpha)
        self.tables_: dict[tuple, np.ndarray] = {}
        self.prior_: np.ndarray | None = None

    @staticmethod
    def contexts_from_example(x: np.ndarray, length: int, token_id_to_text: dict[int, str]) -> tuple:
        class_lookup = {cid: i for i, cid in enumerate(CLASS_IDS)}
        eventsets = []
        for p in range(int(length)):
            labels = set()
            for raw_id in x[p].tolist():
                if int(raw_id) == 0:
                    continue
                token = token_id_to_text.get(int(raw_id), "")
                if token in class_lookup:
                    labels.add(class_lookup[token])
            # Skip events without one of the 12 task labels, as the Markov
            # baseline models transitions among the target-label vocabulary.
            if labels:
                eventsets.append(tuple(sorted(labels)))
        if not eventsets:
            eventsets = [(-1,)]
        return tuple(eventsets[-2:])

    def fit(self, X: np.ndarray, lengths: np.ndarray, y: np.ndarray, token_id_to_text: dict[int, str]) -> "MarkovPredictor":
        prior_counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=self.n_classes).astype(float)
        self.prior_ = (prior_counts + self.alpha) / (prior_counts.sum() + self.alpha * self.n_classes)
        counts: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_classes, dtype=float))
        for x, length, target in zip(X, lengths, y):
            context = self.contexts_from_example(x, int(length), token_id_to_text)
            context = tuple(context[-self.order:])
            counts[context][int(target)] += 1.0
        self.tables_ = dict(counts)
        return self

    def predict_proba(self, X: np.ndarray, lengths: np.ndarray, token_id_to_text: dict[int, str]) -> np.ndarray:
        if self.prior_ is None:
            raise RuntimeError("Call fit first.")
        out = np.zeros((len(X), self.n_classes), dtype=float)
        for idx, (x, length) in enumerate(zip(X, lengths)):
            context = self.contexts_from_example(x, int(length), token_id_to_text)
            context = tuple(context[-self.order:])
            counts = self.tables_.get(context)
            if counts is None:
                out[idx] = self.prior_
            else:
                smoothed = counts + self.alpha * self.prior_
                out[idx] = smoothed / smoothed.sum()
        return out


class EventEncoder(nn.Module):
    """Average the embeddings of co-occurring techniques for each event."""
    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, seq_len, labels_per_event]
        embedded = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1).to(embedded.dtype)
        summed = (embedded * mask).sum(dim=2)
        count = mask.sum(dim=2).clamp_min(1.0)
        event_emb = summed / count
        # All-zero event rows used in ablation are represented by a zero vector.
        event_emb = event_emb * (mask.sum(dim=2) > 0).to(event_emb.dtype)
        return self.dropout(event_emb)


class GRUAttentionModel(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, embedding_dim: int = 32,
                 hidden_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        self.encoder = EventEncoder(vocab_size, embedding_dim, dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.attention_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        x = self.encoder(token_ids)
        hidden_states, _ = self.gru(x)
        score = self.attention_score(torch.tanh(self.attention_hidden(hidden_states))).squeeze(-1)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
        valid = positions < lengths[:, None]
        score = score.masked_fill(~valid, -1e9)
        weights = torch.softmax(score, dim=1)
        context = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(context))
        if return_attention:
            return logits, weights
        return logits


class GRUWithoutAttentionModel(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, embedding_dim: int = 32,
                 hidden_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        self.encoder = EventEncoder(vocab_size, embedding_dim, dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        x = self.encoder(token_ids)
        hidden_states, _ = self.gru(x)
        last_idx = (lengths - 1).clamp_min(0)
        final_state = hidden_states[torch.arange(len(lengths), device=token_ids.device), last_idx]
        logits = self.classifier(self.dropout(final_state))
        if return_attention:
            raise ValueError("GRUWithoutAttentionModel has no attention weights.")
        return logits


class CausalTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, max_length: int = 15,
                 embedding_dim: int = 32, n_heads: int = 4, n_layers: int = 1,
                 feedforward_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.encoder = EventEncoder(vocab_size, embedding_dim, dropout)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True, norm_first=False, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        x = self.encoder(token_ids)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = x + self.position_embedding(positions)[None, :, :]
        seq_len = token_ids.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=token_ids.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = positions[None, :] >= lengths[:, None]
        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        last_idx = (lengths - 1).clamp_min(0)
        last = hidden[torch.arange(len(lengths), device=token_ids.device), last_idx]
        logits = self.classifier(self.dropout(last))
        if return_attention:
            return logits, None
        return logits
