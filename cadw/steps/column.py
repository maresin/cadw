"""ColumnTransformer (moved from cadw/pipeline.py v1.0.0)."""
from __future__ import annotations
from typing import Optional

from cadw.core.models import DataChars
from cadw.pipeline import _merge_fanout, _STEP_REGISTRY


class ColumnTransformer:
    """Apply different transformers to different column subsets.

    sklearn-style column fanout. Each transformer is fit and applied
    to `df[cols]` subset; outputs are concatenated (column union).

    Args:
        transformers: list of (name, cols, step) triples.
            - name: str, unique identifier per transformer.
            - cols: str or list[str] — column subset the transformer sees.
            - step: any Transformer (fit + transform contract).
        remainder: 'drop' (default) | 'passthrough'.
            - 'drop': columns not claimed by any transformer are dropped.
            - 'passthrough': they appear unchanged at the end of output.

    Valid cases:
        - Disjoint `cols` across transformers (each column claimed by ≤1).
        - Transformers preserve row count (no dropna inside).
        - Unique transformer names.
        - `remainder='drop'` with all cols claimed.
        - `remainder='passthrough'` with some cols unclaimed.
        - Transformer emits new columns (e.g. appends `{col}_{feat}`)
          — new cols appear in output.

    Invalid cases (raise at construction):
        - Empty `transformers` → ValueError.
        - Duplicate transformer `name` → ValueError.
        - Overlapping `cols` across transformers → ValueError.
        - Empty `cols` in a transformer → ValueError.
        - `step` missing `fit` or `transform` → TypeError.
        - `remainder` not in ('drop', 'passthrough') → ValueError.

    Invalid cases (raise at transform):
        - `cols` column missing from input df → KeyError.
        - Transformer changed row count (e.g. dropna inside) → RuntimeError.
        - Duplicate output column name across transformers → RuntimeError.

    Output column order:
        [t1 output cols..., t2 output cols..., ..., remainder cols...]
        Where "t_i output cols" = whatever `t_i.transform()` emitted.

    Examples:
        >>> ct = ColumnTransformer([
        ...     ('num', ['a', 'b'], CastStep({'a': 'float64', 'b': 'float64'})),
        ...     ('cat', ['c'], CastStep({'c': 'int32'})),
        ... ], remainder='drop')
        >>> ct.fit(df)
        >>> out = ct.transform(df)  # columns: a, b, c
    """

    __slots__ = ("_transformers", "_remainder")

    def __init__(
        self,
        transformers: list,
        *,
        remainder: str = "drop",
    ) -> None:
        if remainder not in ("drop", "passthrough"):
            raise ValueError(
                f"ColumnTransformer: remainder must be 'drop' or "
                f"'passthrough'; got {remainder!r}"
            )
        if not transformers:
            raise ValueError(
                "ColumnTransformer: transformers must be non-empty list"
            )

        names_seen: set = set()
        cols_claimed: dict = {}  # col → transformer name
        normalized: list = []

        for entry in transformers:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise TypeError(
                    f"ColumnTransformer: each transformer must be "
                    f"(name, cols, step) triple; got {entry!r}"
                )
            name, cols, step = entry

            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"ColumnTransformer: transformer name must be "
                    f"non-empty str; got {name!r}"
                )
            if name in names_seen:
                raise ValueError(
                    f"ColumnTransformer: duplicate transformer name "
                    f"{name!r}; existing={sorted(names_seen)}"
                )
            names_seen.add(name)

            if isinstance(cols, str):
                cols = [cols]
            else:
                cols = list(cols)
            if not cols:
                raise ValueError(
                    f"ColumnTransformer: transformer {name!r}: cols "
                    f"must be non-empty"
                )
            for c in cols:
                if not isinstance(c, str):
                    raise TypeError(
                        f"ColumnTransformer: transformer {name!r}: cols "
                        f"must be str; got {type(c).__name__}"
                    )
                if c in cols_claimed:
                    raise ValueError(
                        f"ColumnTransformer: column {c!r} claimed by "
                        f"both {cols_claimed[c]!r} and {name!r} — each "
                        f"column can belong to at most one transformer"
                    )
                cols_claimed[c] = name

            if not (hasattr(step, "fit") and hasattr(step, "transform")):
                raise TypeError(
                    f"ColumnTransformer: transformer {name!r} "
                    f"({type(step).__name__}) must have fit + transform "
                    f"(Transformer contract); Predictors cannot be "
                    f"inside ColumnTransformer (only as final Pipeline step)"
                )

            normalized.append((name, cols, step))

        self._transformers = normalized
        self._remainder = remainder

    def fit(self, df, target=None):
        """Fit each transformer on its column subset."""
        for name, cols, step in self._transformers:
            for c in cols:
                if c not in df.columns:
                    raise KeyError(
                        f"ColumnTransformer.fit: transformer {name!r}: "
                        f"column {c!r} not in {df.columns}"
                    )
            subset = df[cols]
            step.fit(subset, target=None)
        return self

    def transform(self, df):
        """Apply each transformer to its subset; column-union output."""
        n_orig = df.count()
        claimed = set()
        for _, cols, _ in self._transformers:
            claimed.update(cols)

        results = []
        out_col_owner: dict = {}  # col → transformer name (for duplicate check)

        for name, cols, step in self._transformers:
            for c in cols:
                if c not in df.columns:
                    raise KeyError(
                        f"ColumnTransformer.transform: transformer "
                        f"{name!r}: column {c!r} not in {df.columns}"
                    )
            subset = df[cols]
            out = step.transform(subset)
            n_out = out.count()
            if n_out != n_orig:
                raise RuntimeError(
                    f"ColumnTransformer: transformer {name!r} changed "
                    f"row count: {n_orig} -> {n_out}. ColumnTransformer "
                    f"requires row-preserving transformers (column-level "
                    f"only). If you need row filtering, apply it before "
                    f"ColumnTransformer."
                )
            for c in out.columns:
                if c in out_col_owner:
                    raise RuntimeError(
                        f"ColumnTransformer: duplicate output column "
                        f"{c!r} from transformers "
                        f"{out_col_owner[c]!r} and {name!r}. Use distinct "
                        f"column names or check for overlapping transforms."
                    )
                out_col_owner[c] = name
            results.append(out)

        if self._remainder == "passthrough":
            remainder_cols = [c for c in df.columns if c not in claimed]
            if remainder_cols:
                results.append(df[remainder_cols])

        return _merge_fanout(results)

    def estimate_state_bytes(self, dc) -> int:
        return sum(
            step.estimate_state_bytes(dc)
            for _, _, step in self._transformers
        )

    def state_bytes(self) -> int:
        return sum(
            step.state_bytes()
            for _, _, step in self._transformers
        )

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (nested state inline)."""
        entries = []
        for name, cols, step in self._transformers:
            if not (hasattr(step, "to_dict") and callable(step.to_dict)):
                raise NotImplementedError(
                    f"ColumnTransformer.to_dict: nested step {name!r} "
                    f"({type(step).__name__}) does not implement to_dict()"
                )
            # v0.9.9 (N1): use __serialization_name__ (set by
            # register_step_type) so from_json finds the same key.
            entries.append({
                "name": name,
                "cols": list(cols),
                "type": getattr(
                    type(step), "__serialization_name__",
                    type(step).__name__,
                ),
                "step_dict": step.to_dict(),
            })
        return {
            "transformers": entries,
            "remainder": self._remainder,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnTransformer":
        entries = []
        for entry in d.get("transformers", []):
            type_name = entry["type"]
            if type_name not in _STEP_REGISTRY:
                raise NotImplementedError(
                    f"ColumnTransformer.from_dict: unknown nested step "
                    f"type {type_name!r}. Register via "
                    f"register_step_type({type_name!r}, ...) or use "
                    f"built-in wrappers: {list_step_types()}"
                )
            step_cls = _STEP_REGISTRY[type_name]
            step = step_cls.from_dict(entry["step_dict"])
            entries.append((entry["name"], entry["cols"], step))
        return cls(entries, remainder=d.get("remainder", "drop"))

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{name}={list(cols)}" for name, cols, _ in self._transformers
        )
        return (
            f"ColumnTransformer([{parts}], remainder={self._remainder!r})"
        )
