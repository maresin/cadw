# cadw/dataframe/_columns.py
# v0.9.26x (B.1 step 18): _ColumnsMixin.
# Symbol: select_dtypes (missed from v0.9.26j C-1 batch).
# Extracted verbatim from _core.py; no logic change.

from __future__ import annotations

import pyarrow as pa


class _ColumnsMixin:
    """Column selection mixin (v0.9.26x)."""
    def select_dtypes(
        self,
        include=None,
        exclude=None,
    ) -> "CadwDataFrame":
        """Select columns by dtype predicate (v0.9.25g).

        - include / exclude: dtype spec (str or iterable of str).
          At least one required.
        - Keep column if (matches any include OR include is None/'all')
          AND (matches no exclude).
        - Dtype specs: 'number', 'int'/'integer', 'float'/'floating',
          'bool'/'boolean', 'string'/'object', 'datetime',
          'category', 'all', or explicit pyarrow type name
          (e.g. 'int64', 'float64', 'string').
        - Returns new CadwDataFrame with selected columns;
          filters preserved.
        - No-match → ValueError.

        Example:
            df.select_dtypes(include='number')
            df.select_dtypes(exclude=['string', 'bool'])
            df.select_dtypes(include='number', exclude='float32')
        """
        if include is None and exclude is None:
            raise ValueError(
                "select_dtypes: at least one of include/exclude required"
            )

        def _normalize(spec):
            if spec is None:
                return None
            if isinstance(spec, str):
                return [spec]
            return list(spec)

        def _matches(t: pa.DataType, s: str) -> bool:
            if s == "all":
                return True
            if s == "number":
                return pa.types.is_integer(t) or pa.types.is_floating(t)
            if s in ("int", "integer"):
                return pa.types.is_integer(t)
            if s in ("float", "floating"):
                return pa.types.is_floating(t)
            if s in ("bool", "boolean"):
                return pa.types.is_boolean(t)
            if s in ("string", "object"):
                return (
                    pa.types.is_string(t) or pa.types.is_large_string(t)
                    or pa.types.is_binary(t) or pa.types.is_large_binary(t)
                )
            if s in ("datetime", "datetime64"):
                return pa.types.is_timestamp(t)
            if s == "category":
                return pa.types.is_dictionary(t)
            # v0.9.25g-fix1 (§13.4/262): PyArrow dtype reprs differ
            # from user-facing names. str(pa.float32()) == "float",
            # str(pa.float64()) == "double". Use type_for_alias first
            # (same resolver astype uses), fallback to str-compare.
            try:
                return t == pa.type_for_alias(s)
            except (ValueError, KeyError, TypeError):
                return str(t) == s

        inc_list = _normalize(include)
        exc_list = _normalize(exclude)

        keep = []
        for c in self._columns:
            t = self._dataset.schema.field(c).type
            if inc_list is not None and "all" not in inc_list:
                if not any(_matches(t, s) for s in inc_list):
                    continue
            if exc_list is not None:
                if any(_matches(t, s) for s in exc_list):
                    continue
            keep.append(c)

        if not keep:
            raise ValueError(
                f"select_dtypes: no columns matched "
                f"include={include!r} exclude={exclude!r}; "
                f"available dtypes={self.dtypes()}"
            )
        return self[keep]

