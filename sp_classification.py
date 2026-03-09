"""
sp_classification.py
====================
Signal Peptide Classifier — Object-Oriented Pipeline
=====================================================

Pipeline:
    1. KaggleSignalPDatasetLoader  — loads FASTA / TSV files, filters, splits
    2. ESM2Embedder                — converts sequences → 480-dim ESM-2 embeddings
    3. CNNClassifier               — 1D-CNN on embeddings
       LSTMClassifier              — Bidirectional LSTM on embeddings
       TransformerClassifier       — Transformer encoder on embeddings
    4. ProteinDataset              — PyTorch Dataset wrapper
    5. ModelTrainer                — trains a single model with early stopping
    6. ModelEvaluator              — computes Accuracy / Precision / Recall / F1 / AUC
    7. SignalPeptidePredictor      — inference wrapper for new sequences
    8. Visualizer                  — bar-chart, ranking, ROC curves
    9. SignalPeptidePipeline       — orchestrates the full end-to-end run

Usage (standalone):
    python sp_classification.py

Usage (import):
    from sp_classification import SignalPeptidePipeline
    pipeline = SignalPeptidePipeline(data_dir="/kaggle/input/signal-peptide")
    pipeline.run()
    pipeline.predict(["MKKLLFVLLFVLLVSSAYSR"])
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library
# ──────────────────────────────────────────────────────────────────────────────
import copy
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

class KaggleSignalPDatasetLoader:
    """
    Load the Kaggle Signal Peptide dataset from FASTA and TSV files.

    Expected files in *data_dir*:
        positive.fasta, negative.fasta
        positive.tsv,   negative.tsv

    Parameters
    ----------
    data_dir : str
        Directory containing the four dataset files.
    max_seq_len : int
        Sequences longer than this are discarded (default 70).
    test_size : float
        Fraction of data held out for testing (default 0.20).
    random_state : int
        Seed for reproducible train/test split (default 42).
    """

    VALID_AA: frozenset = frozenset("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        data_dir: str = "/kaggle/input/signal-peptide",
        max_seq_len: int = 70,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_fasta(self, filepath: Path, label: int) -> List[Tuple[str, str, int]]:
        """Parse a FASTA file and return (header, sequence, label) triples."""
        records: List[Tuple[str, str, int]] = []
        header: Optional[str] = None
        seq_parts: List[str] = []

        with open(filepath, "r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header and seq_parts:
                        records.append((header, "".join(seq_parts), label))
                    header = line[1:]
                    seq_parts = []
                else:
                    seq_parts.append(line)

        if header and seq_parts:
            records.append((header, "".join(seq_parts), label))

        return records

    def _load_tsv(self, filepath: Path, label: int) -> List[Tuple[str, str, int]]:
        """Parse a TSV file; auto-detect the sequence column."""
        df = pd.read_csv(filepath, sep="\t")
        seq_cols = [c for c in df.columns if "seq" in c.lower() or "sequence" in c.lower()]
        if not seq_cols:
            raise ValueError(
                f"No sequence column found in {filepath}. "
                f"Columns present: {list(df.columns)}"
            )
        seq_col = seq_cols[0]
        return [("", str(row[seq_col]).strip(), label) for _, row in df.iterrows()]

    def _filter(self, records: List[Tuple]) -> List[Tuple]:
        """Discard sequences that are too short, too long, or contain non-standard residues."""
        return [
            (h, s.upper(), lbl)
            for h, s, lbl in records
            if 5 <= len(s) <= self.max_seq_len and set(s.upper()).issubset(self.VALID_AA)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Dict]:
        """
        Load, filter, and split the dataset.

        Returns
        -------
        dict
            ``{"train": {"sequences": [...], "labels": np.ndarray},
               "test":  {"sequences": [...], "labels": np.ndarray}}``
        """
        print("📂  Loading Kaggle signal-peptide dataset …")

        pos = self._filter(
            self._load_fasta(self.data_dir / "positive.fasta", label=1)
            + self._load_tsv(self.data_dir / "positive.tsv", label=1)
        )
        neg = self._filter(
            self._load_fasta(self.data_dir / "negative.fasta", label=0)
            + self._load_tsv(self.data_dir / "negative.tsv", label=0)
        )

        print(f"   • Positive (signal peptide) : {len(pos):,}")
        print(f"   • Negative (no signal peptide): {len(neg):,}")

        all_records = pos + neg
        np.random.shuffle(all_records)

        _, sequences, labels = zip(*all_records)
        sequences = list(sequences)
        labels = np.array(labels)

        tr_seqs, te_seqs, tr_lbls, te_lbls = train_test_split(
            sequences,
            labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state,
        )

        print(f"\n📊  Split  →  train: {len(tr_seqs):,}  |  test: {len(te_seqs):,}")

        return {
            "train": {"sequences": tr_seqs, "labels": tr_lbls},
            "test":  {"sequences": te_seqs, "labels": te_lbls},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ESM-2 EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════

class ESM2Embedder:
    """
    Convert raw amino-acid sequences to dense vectors using ESM-2.

    The default model ``facebook/esm2_t12_35M_UR50D`` produces
    **480-dimensional** embeddings after mean-pooling over residue positions.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for ESM-2.
    device : str or None
        ``"cuda"`` / ``"cpu"``; auto-detected when *None*.
    pool_mode : str
        ``"mean"`` (masked mean-pool) or ``"cls"`` (first token).
    batch_size : int
        Sequences processed per forward pass (default 16).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        device: Optional[str] = None,
        pool_mode: str = "mean",
        batch_size: int = 16,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pool_mode = pool_mode
        self.batch_size = batch_size

        print(f"🚀  Loading ESM-2 ({model_name}) on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("✅  ESM-2 ready")

    @torch.no_grad()
    def embed(self, sequences: List[str]) -> np.ndarray:
        """
        Embed a list of amino-acid sequences.

        Parameters
        ----------
        sequences : list[str]
            Raw sequences (no spaces needed — handled internally).

        Returns
        -------
        np.ndarray
            Shape ``(N, embedding_dim)``.
        """
        all_embs: List[np.ndarray] = []

        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Embedding"):
            batch = [" ".join(s) for s in sequences[i : i + self.batch_size]]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

            hidden = self.model(**inputs).last_hidden_state  # (B, L, D)

            if self.pool_mode == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)   # (B, D)
            else:  # cls
                pooled = hidden[:, 0, :]  # (B, D)

            all_embs.append(pooled.cpu().numpy())

            if i % (self.batch_size * 10) == 0 and self.device == "cuda":
                torch.cuda.empty_cache()

        return np.vstack(all_embs)

    def free(self) -> None:
        """Release model weights from GPU memory."""
        del self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NEURAL NETWORK ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class CNNClassifier(nn.Module):
    """
    1-D Convolutional classifier operating on pooled protein embeddings.

    Architecture
    ------------
    ``[B, D] → unsqueeze → Conv1d(D→256,k=3) → Conv1d(256→128,k=3)
             → AdaptiveMaxPool → FC(128→64) → Dropout → FC(64→2)``

    Parameters
    ----------
    input_dim : int
        Embedding dimension (480 for ESM-2 t12).
    num_classes : int
        Number of output classes (default 2).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = 480,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1   = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveMaxPool1d(1)
        self.fc1     = nn.Linear(128, 64)
        self.fc2     = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)                          # (B, D, 1)
        x = F.relu(self.conv1(x))                    # (B, 256, 1)
        x = F.relu(self.conv2(x))                    # (B, 128, 1)
        x = self.pool(x).squeeze(-1)                 # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))        # (B, 64)
        return self.fc2(x)                           # (B, num_classes)


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier operating on pooled protein embeddings.

    Architecture
    ------------
    ``[B, D] → unsqueeze(1) → BiLSTM(layers=2) → last hidden
             → Dropout → FC(hidden→2)``

    Parameters
    ----------
    input_dim : int
        Embedding dimension.
    hidden_dim : int
        LSTM hidden state size (default 256).
    num_layers : int
        Stacked LSTM layers (default 2).
    num_classes : int
        Output classes (default 2).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = 480,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc      = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)             # (B, 1, D)  — treat embedding as length-1 sequence
        _, (hn, _) = self.lstm(x)      # hn: (layers, B, hidden)
        x = self.dropout(hn[-1])       # last layer hidden state: (B, hidden)
        return self.fc(x)              # (B, num_classes)


class TransformerClassifier(nn.Module):
    """
    Transformer-encoder classifier operating on pooled protein embeddings.

    Architecture
    ------------
    ``[B, D] → unsqueeze(1) + learnable pos-enc
             → TransformerEncoder(layers=2, heads=8)
             → cls token → Dropout → FC(D→2)``

    Parameters
    ----------
    input_dim : int
        Embedding dimension.
    num_heads : int
        Multi-head attention heads (default 8).
    num_layers : int
        Stacked encoder layers (default 2).
    num_classes : int
        Output classes (default 2).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = 480,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc           = nn.Linear(input_dim, num_classes)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) + self.pos_encoding   # (B, 1, D)
        x = self.transformer(x)                  # (B, 1, D)
        x = self.dropout(x[:, 0, :])             # first token: (B, D)
        return self.fc(x)                        # (B, num_classes)


# ──────────────────────────────────────────────────────────────────────────────
# Factory — convenient ``build_models()`` helper
# ──────────────────────────────────────────────────────────────────────────────

def build_models(input_dim: int = 480, device: str = "cpu") -> Dict[str, nn.Module]:
    """
    Instantiate all three classifiers and move them to *device*.

    Returns
    -------
    dict
        ``{"CNN": model, "LSTM": model, "Transformer": model}``
    """
    return {
        "CNN":         CNNClassifier(input_dim=input_dim).to(device),
        "LSTM":        LSTMClassifier(input_dim=input_dim).to(device),
        "Transformer": TransformerClassifier(input_dim=input_dim).to(device),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ProteinDataset(Dataset):
    """
    Minimal PyTorch ``Dataset`` wrapping embedding arrays and labels.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, embedding_dim)``.
    y : np.ndarray
        Integer label array of shape ``(N,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Train a single PyTorch model with early stopping.

    Parameters
    ----------
    model : nn.Module
        Uninitialised (fresh) model instance on the correct device.
    device : str
        ``"cuda"`` or ``"cpu"``.
    lr : float
        AdamW learning rate (default ``1e-4``).
    weight_decay : float
        AdamW weight decay (default ``1e-5``).
    num_epochs : int
        Maximum training epochs (default 30).
    patience : int
        Early-stopping patience in epochs (default 5).
    batch_size : int
        Training mini-batch size (default 32).
    val_fraction : float
        Fraction of training data used for validation (default 0.10).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 30,
        patience: int = 5,
        batch_size: int = 32,
        val_fraction: float = 0.10,
    ) -> None:
        self.model        = model
        self.device       = device
        self.num_epochs   = num_epochs
        self.patience     = patience
        self.batch_size   = batch_size
        self.val_fraction = val_fraction
        self.criterion    = nn.CrossEntropyLoss()
        self.optimizer    = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": []
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_loaders(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        val_n = max(1, int(self.val_fraction * len(X)))
        X_tr, X_val = X[val_n:], X[:val_n]
        y_tr, y_val = y[val_n:], y[:val_n]

        train_loader = DataLoader(
            ProteinDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            ProteinDataset(X_val, y_val),
            batch_size=64,
            shuffle=False,
        )
        return train_loader, val_loader

    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        """Single forward (+ optional backward) pass; returns (loss, accuracy)."""
        self.model.train(training)
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                out  = self.model(X_b)
                loss = self.criterion(out, y_b)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(y_b)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc      = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ModelTrainer":
        """
        Train the model.

        Parameters
        ----------
        X_train : np.ndarray
            Training embeddings ``(N, D)``.
        y_train : np.ndarray
            Integer labels ``(N,)``.

        Returns
        -------
        self
        """
        train_loader, val_loader = self._make_loaders(X_train, y_train)

        best_val_loss  = float("inf")
        best_weights   = None
        patience_count = 0

        for epoch in range(1, self.num_epochs + 1):
            tr_loss, _      = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"  Epoch {epoch:>3}/{self.num_epochs} │ "
                f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = copy.deepcopy(self.model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f"  ⏹  Early stopping at epoch {epoch}")
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    Compute classification metrics on a held-out test set.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    device : str
        Computation device.
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model  = model
        self.device = device

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Evaluate the model and return a metrics dict.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``auc``.
        """
        loader = DataLoader(
            ProteinDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        self.model.eval()
        all_preds: List[int]   = []
        all_probs: List[float] = []
        all_labels: List[int]  = []

        with torch.no_grad():
            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                out  = self.model(X_b)
                prob = F.softmax(out, dim=1)

                all_preds.extend(out.argmax(1).cpu().numpy())
                all_probs.extend(prob[:, 1].cpu().numpy())   # P(signal peptide)
                all_labels.extend(y_b.numpy())

        acc           = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary"
        )
        auc = (
            roc_auc_score(all_labels, all_probs)
            if len(np.unique(all_labels)) > 1
            else 0.0
        )

        return {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "auc":       auc,
        }

    def get_roc_data(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (fpr, tpr) arrays for ROC curve plotting.
        """
        loader = DataLoader(
            ProteinDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        self.model.eval()
        all_probs:  List[float] = []
        all_labels: List[int]   = []

        with torch.no_grad():
            for X_b, y_b in loader:
                X_b  = X_b.to(self.device)
                prob = F.softmax(self.model(X_b), dim=1)
                all_probs.extend(prob[:, 1].cpu().numpy())
                all_labels.extend(y_b.numpy())

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        return fpr, tpr


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PREDICTOR (Inference Wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class SignalPeptidePredictor:
    """
    Deployment-ready inference wrapper.

    Accepts raw amino-acid sequences, embeds them on-the-fly with ESM-2,
    and returns predicted classes + confidence scores.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    embedder : ESM2Embedder
        Loaded embedder instance.
    device : str
        Computation device.
    """

    VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        model: nn.Module,
        embedder: ESM2Embedder,
        device: str = "cpu",
    ) -> None:
        self.model    = model.to(device).eval()
        self.embedder = embedder
        self.device   = device

    @torch.no_grad()
    def predict(
        self,
        sequences: List[str],
        return_probs: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict signal-peptide presence for a batch of sequences.

        Parameters
        ----------
        sequences : list[str]
            Raw amino-acid sequences (uppercase or lowercase).
        return_probs : bool
            If ``True``, include per-sequence confidence in the output.

        Returns
        -------
        dict
            ``{"predictions": np.ndarray, "probabilities": np.ndarray}``
            *predictions* — 0 (no signal) / 1 (signal peptide).
            *probabilities* — P(signal peptide).
        """
        # --- Input validation ---
        clean: List[str] = []
        for i, seq in enumerate(sequences):
            s = seq.upper().strip()
            invalid = set(s) - self.VALID_AA
            if invalid:
                raise ValueError(
                    f"Sequence {i} contains invalid residues: {invalid}"
                )
            if len(s) > 70:
                print(f"⚠️  Sequence {i} trimmed to 70 residues (was {len(s)})")
                s = s[:70]
            clean.append(s)

        # --- Embed & predict ---
        X     = self.embedder.embed(clean)
        X_t   = torch.tensor(X, dtype=torch.float32).to(self.device)
        out   = self.model(X_t)
        prob  = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()

        result = {"predictions": preds}
        if return_probs:
            result["probabilities"] = prob

        return result

    def predict_single(self, sequence: str) -> str:
        """Convenience method — returns a human-readable label for one sequence."""
        out = self.predict([sequence])
        label = "✅ SIGNAL PEPTIDE" if out["predictions"][0] == 1 else "❌ NO SIGNAL PEPTIDE"
        conf  = out["probabilities"][0]
        return f"{label}  (confidence: {conf:.2%})"


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Generate and (optionally) save performance plots.

    Parameters
    ----------
    save_dir : str or None
        If provided, plots are saved as PNG files to this directory.
    """

    def __init__(self, save_dir: Optional[str] = None) -> None:
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_or_show(self, filename: str) -> None:
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=150, bbox_inches="tight")
            print(f"   💾 Saved → {self.save_dir / filename}")
        plt.show()
        plt.close()

    # ------------------------------------------------------------------

    def plot_metric_comparison(
        self, results: Dict[str, Dict]
    ) -> None:
        """Bar chart comparing all metrics across models."""
        rows = [
            {"Model": name, "Metric": m.upper(), "Value": v}
            for name, res in results.items()
            for m, v in res["test_metrics"].items()
        ]
        df = pd.DataFrame(rows)

        plt.figure(figsize=(14, 7))
        sns.barplot(
            data=df,
            x="Metric",
            y="Value",
            hue="Model",
            hue_order=list(results.keys()),
        )
        plt.title("Model Performance Comparison", fontsize=15, fontweight="bold")
        plt.ylabel("Score")
        plt.ylim(0, 1.08)
        plt.xticks(rotation=30)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        self._save_or_show("metric_comparison.png")

    def plot_ranking(self, results: Dict[str, Dict], metric: str = "f1") -> None:
        """Horizontal bar chart ranking models by *metric*."""
        sorted_items = sorted(
            results.items(),
            key=lambda kv: kv[1]["test_metrics"][metric],
            reverse=True,
        )
        names  = [k for k, _ in sorted_items]
        scores = [v["test_metrics"][metric] for _, v in sorted_items]
        colors = ["#4c72b0", "#55a868", "#c44e52"][: len(names)]

        plt.figure(figsize=(9, 5))
        bars = plt.barh(names[::-1], scores[::-1], color=colors[::-1])
        for bar, score in zip(bars, scores[::-1]):
            plt.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center",
                fontweight="bold",
            )
        plt.xlabel(metric.upper() + " Score")
        plt.title(f"Model Ranking by {metric.upper()}", fontsize=14, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        self._save_or_show("model_ranking.png")

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        results:  Dict[str, Dict],
    ) -> None:
        """Overlay ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        for name, (fpr, tpr) in roc_data.items():
            auc = results[name]["test_metrics"]["auc"]
            plt.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {auc:.4f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random  (AUC = 0.5000)")
        plt.xlim([0, 1])
        plt.ylim([0, 1.02])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(linestyle="--", alpha=0.5)
        plt.tight_layout()
        self._save_or_show("roc_curves.png")

    @staticmethod
    def print_summary(results: Dict[str, Dict]) -> None:
        """Print a formatted metrics table to stdout."""
        rows = []
        for name, res in results.items():
            row = {"Model": name}
            row.update({k.upper(): f"{v:.4f}" for k, v in res["test_metrics"].items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        print("\n" + "═" * 72)
        print("  FINAL MODEL COMPARISON")
        print("═" * 72)
        print(df.to_string(index=False))

        best = max(results, key=lambda k: results[k]["test_metrics"]["f1"])
        best_f1 = results[best]["test_metrics"]["f1"]
        print(f"\n🏆  Best model  →  {best}  (F1 = {best_f1:.4f})")
        print("═" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class SignalPeptidePipeline:
    """
    End-to-end orchestrator for the signal-peptide classification project.

    Stages
    ------
    1. Load dataset  → ``KaggleSignalPDatasetLoader``
    2. Embed         → ``ESM2Embedder``
    3. Build models  → ``build_models()``
    4. Train + eval  → ``ModelTrainer`` + ``ModelEvaluator``
    5. Visualise     → ``Visualizer``

    Parameters
    ----------
    data_dir : str
        Directory containing the Kaggle dataset files.
    model_name : str
        ESM-2 HuggingFace identifier.
    max_seq_len : int
        Maximum sequence length kept after filtering.
    num_epochs : int
        Max training epochs per model.
    patience : int
        Early-stopping patience.
    save_dir : str or None
        If set, plots and results are saved here.
    """

    def __init__(
        self,
        data_dir: str = "/kaggle/input/signal-peptide",
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        max_seq_len: int = 70,
        num_epochs: int = 15,
        patience: int = 3,
        save_dir: Optional[str] = None,
    ) -> None:
        self.data_dir   = data_dir
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.num_epochs  = num_epochs
        self.patience    = patience
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"

        self.visualizer = Visualizer(save_dir=save_dir)

        # State filled in by run()
        self._embedded: Optional[Dict]         = None
        self._models:   Optional[Dict]         = None
        self._results:  Optional[Dict]         = None
        self._best_model_name: Optional[str]   = None
        self._embedder: Optional[ESM2Embedder] = None

    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Dict]:
        """
        Execute the full pipeline and return the results dict.

        Returns
        -------
        dict
            ``{model_name: {"model": nn.Module, "test_metrics": {...}}}``
        """
        # ── 1. Load data ──────────────────────────────────────────────
        loader = KaggleSignalPDatasetLoader(
            data_dir=self.data_dir, max_seq_len=self.max_seq_len
        )
        data = loader.load()

        # ── 2. Embed ──────────────────────────────────────────────────
        self._embedder = ESM2Embedder(model_name=self.model_name)
        embedded: Dict[str, Dict] = {}
        for split, content in data.items():
            print(f"\n[{split.upper()}] Embedding {len(content['sequences']):,} sequences …")
            X = self._embedder.embed(content["sequences"])
            embedded[split] = {"X": X, "y": content["labels"]}
            print(f"   → shape: {X.shape}")

        self._embedded = embedded
        input_dim = embedded["train"]["X"].shape[1]

        # ── 3. Build models ───────────────────────────────────────────
        self._models = build_models(input_dim=input_dim, device=self.device)

        # ── 4. Train + evaluate ───────────────────────────────────────
        results: Dict[str, Dict] = {}
        roc_data: Dict[str, Tuple] = {}

        X_tr, y_tr = embedded["train"]["X"], embedded["train"]["y"]
        X_te, y_te = embedded["test"]["X"],  embedded["test"]["y"]

        for name, model in self._models.items():
            print(f"\n{'═'*60}")
            print(f"  TRAINING  {name}")
            print("═" * 60)

            trainer = ModelTrainer(
                model=model,
                device=self.device,
                num_epochs=self.num_epochs,
                patience=self.patience,
            )
            trainer.fit(X_tr, y_tr)

            evaluator = ModelEvaluator(model=model, device=self.device)
            metrics   = evaluator.evaluate(X_te, y_te)
            fpr, tpr  = evaluator.get_roc_data(X_te, y_te)

            print(f"\n  ✅  {name} test metrics:")
            for k, v in metrics.items():
                print(f"       {k.upper():<12} {v:.4f}")

            results[name]  = {"model": model, "test_metrics": metrics}
            roc_data[name] = (fpr, tpr)

        self._results = results
        self._best_model_name = max(results, key=lambda k: results[k]["test_metrics"]["f1"])

        # ── 5. Visualise ──────────────────────────────────────────────
        self.visualizer.print_summary(results)
        self.visualizer.plot_metric_comparison(results)
        self.visualizer.plot_ranking(results, metric="f1")
        self.visualizer.plot_roc_curves(roc_data, results)

        return results

    # ------------------------------------------------------------------

    def predict(
        self,
        sequences: List[str],
        model_name: Optional[str] = None,
    ) -> None:
        """
        Run inference on new sequences and print results.

        Parameters
        ----------
        sequences : list[str]
            Amino-acid sequences to classify.
        model_name : str or None
            Which trained model to use; defaults to the best by F1.
        """
        if self._results is None:
            raise RuntimeError("Call .run() before .predict().")

        chosen = model_name or self._best_model_name
        if self._embedder is None:
            self._embedder = ESM2Embedder(model_name=self.model_name)

        predictor = SignalPeptidePredictor(
            model=self._results[chosen]["model"],
            embedder=self._embedder,
            device=self.device,
        )

        print(f"\n{'═'*60}")
        print(f"  INFERENCE  (model: {chosen})")
        print("═" * 60)

        out = predictor.predict(sequences, return_probs=True)
        for i, (seq, pred, prob) in enumerate(
            zip(sequences, out["predictions"], out["probabilities"])
        ):
            label = "✅ SIGNAL PEPTIDE" if pred == 1 else "❌ NO SIGNAL"
            print(f"  {i+1}. {label}  ({prob:.2%})  │  {seq[:25]}{'…' if len(seq)>25 else ''}")

    def save_results(self, path: str = "results.json") -> None:
        """
        Persist test metrics to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        if self._results is None:
            raise RuntimeError("Call .run() first.")

        payload = {
            name: res["test_metrics"]
            for name, res in self._results.items()
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

        print(f"📄  Metrics saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = SignalPeptidePipeline(
        data_dir="/kaggle/input/signal-peptide",
        num_epochs=15,
        patience=3,
        save_dir="outputs",
    )

    pipeline.run()

    pipeline.predict(
        sequences=[
            "MKKLLFVLLFVLLVSSAYSR",   # E. coli signal peptide
            "MALWMRLLPLLALLALWGP",    # Human preproinsulin SP
            "MAEGEITTFTALTEKFNLPP",   # Non-signal-peptide control
            "MKVLLALVLLAASGASASAQ",   # Archaeal signal peptide
        ]
    )

    pipeline.save_results("results.json")
