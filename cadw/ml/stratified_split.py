"""StratifiedSplit — thin wrapper over cadw.cv.KFold (v0.9.143-batch5).

Promoted from .pingis/impl/O/stratified_split.py (DEC-058) — but
**fixed**: uses verified KFold(stratify_by=...) instead of impl's
_annotate_roothash, which had a rng-in-loop bug (same seed per batch
-> same random values across equally-sized batches -> correlated folds).

State = O(1) params (n_splits, stratify_col, seed) + 128 B header.
bounded_state: true. KFold.streams fold assignment via splitmix64(
row_id, seed) % k — deterministic per row, no persistent state.

YAML: O_eda.yaml::StratifiedSplit. depends_on: [cadw.cv.KFold].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from cadw.core.memory_guard import MemoryGuard
from cadw.cv import KFold


_HEADER_BYTES = 128


@dataclass
class StratifiedSplitModel:
    n_splits: int
    stratify_col: str
    seed: int

    def state_nbytes(self) -> int:
        return int(_HEADER_BYTES)

    def split(self, df, fold: int) -> tuple:
        """Yield (train_df, test_df) for the given fold (0-indexed).

        Delegates to KFold.split — streaming, O(batch) RAM.
        """
        if not (0 <= fold < self.n_splits):
            raise IndexError(
                f"fold={fold} out of range [0,{self.n_splits})")
        cv = KFold(
            n_splits=self.n_splits,
            stratify_by=self.stratify_col,
            random_state=self.seed,
        )
        for fid, train, test in cv.split(df):
            if fid == fold:
                return train, test
        raise RuntimeError(
            f"KFold didn't emit fold {fold}; check df has ≥ n_splits rows")

    def all_splits(self, df) -> Iterator[tuple]:
        """Yield (fold_id, train_df, test_df) for all folds (streaming)."""
        cv = KFold(
            n_splits=self.n_splits,
            stratify_by=self.stratify_col,
            random_state=self.seed,
        )
        yield from cv.split(df)


def fit_stratified_split(
    df, *,
    stratify_col: str,
    n_splits: int = 5,
    seed: int = 42,
    memory_guard=None,
) -> StratifiedSplitModel:
    """Fit a stratified splitter (thin wrapper over KFold).

    Validates stratify_col exists and n_splits >= 2. Does NOT eagerly
    compute fold assignments — split() streams them from KFold.
    O(1) state, O(batch) per split call.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not isinstance(stratify_col, str) or not stratify_col:
        raise ValueError(
            f"stratify_col must be non-empty str, got {stratify_col!r}")
    if stratify_col not in df.columns:
        raise KeyError(
            f"stratify_col {stratify_col!r} not in {list(df.columns)}")
    mg = memory_guard or MemoryGuard(max_memory_mb=512)
    # state O(1); check the header at least (harmless, satisfies G2 spirit)
    mg.check_state(_HEADER_BYTES, name="stratified_split.state")
    return StratifiedSplitModel(
        n_splits=int(n_splits),
        stratify_col=str(stratify_col),
        seed=int(seed),
    )
