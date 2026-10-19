"""
JoinMemoryModel (§9.2.5).

Grace hash join: build = меньшая (inner) или right (left), probe = другая.

    f_join = build_partition_bytes              # materialized build per partition
           + probe_batch_bytes                  # batch_size × row_size_probe
           + output_batch_bytes                 # batch_size × avg_matches × row_size_output
           + c_merge × output_batch_bytes       # pandas temp in pd.merge

`build_partition_bytes` — state, сохраняется на всё время обработки пары
партиций. `probe_batch_bytes` + `output_batch_bytes` — пик на один шаг
`pd.merge`; суммируются с build.

`solve(available, left_chars, op_context={"right_chars": ..., "how": ...})`:
  - `P` = max(shuffle.solve(left), shuffle.solve(right))
  - `build_side` = "right" если how=="left", иначе "left"/"right" (меньший state)
  - budget split: build_ratio=0.4, probe_ratio=0.3, output_ratio=0.3
  - `max_output_rows` = output_budget / (row_size_out × (1 + c_merge))
  - `probe_batch_size` = min(probe_budget/row_size_probe,
                            output_budget / (avg_matches × row_size_out × (1+c)))
  - None если build_partition > build_budget при любом P
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cadw.core.models import (
    DataChars,
    OpMemoryModel,
    Params,
    get_model,
    register_model,
)
from cadw.core.models_shuffle import HashShuffleMemoryModel


@dataclass
class JoinParams(Params):
    build_side: str         # "left" | "right"
    how: str                # "inner" | "left"
    num_partitions: int
    probe_batch_size: int
    build_partition_bytes: int
    max_output_rows: int
    right_chars: DataChars
    probe_row_size: int
    output_row_size: int
    avg_matches: float


class JoinMemoryModel(OpMemoryModel):
    op_name = "join"

    def __init__(
        self,
        shuffle_model: Optional[HashShuffleMemoryModel] = None,
        *,
        build_ratio: float = 0.4,       # CALIBRATION DEFAULT (§9.5)
        probe_ratio: float = 0.3,       # CALIBRATION DEFAULT
        output_ratio: float = 0.3,      # CALIBRATION DEFAULT
        c_merge_ratio: float = 2.0,     # pandas temp ≈ 2 × output
    ) -> None:
        self.shuffle_model = shuffle_model or get_model("hash_shuffle")
        if not isinstance(self.shuffle_model, HashShuffleMemoryModel):
            raise TypeError("shuffle_model must be HashShuffleMemoryModel")
        for name, r in (
            ("build_ratio", build_ratio),
            ("probe_ratio", probe_ratio),
            ("output_ratio", output_ratio),
        ):
            if not (0 < r < 1):
                raise ValueError(f"{name} must be in (0,1), got {r}")
        if build_ratio + probe_ratio + output_ratio > 1.0:
            raise ValueError(
                "build_ratio + probe_ratio + output_ratio must be <= 1.0; "
                f"got {build_ratio} + {probe_ratio} + {output_ratio}"
            )
        if c_merge_ratio < 0:
            raise ValueError("c_merge_ratio must be >= 0")
        self.build_ratio = float(build_ratio)
        self.probe_ratio = float(probe_ratio)
        self.output_ratio = float(output_ratio)
        self.c_merge_ratio = float(c_merge_ratio)

    def estimate(self, params: JoinParams, data_chars: DataChars) -> int:
        """Пик RAM: build_partition + probe_batch + output_batch×(1+c_merge)."""
        if not isinstance(params, JoinParams):
            raise TypeError(f"expected JoinParams, got {type(params).__name__}")
        return int(
            params.build_partition_bytes
            + params.probe_batch_size * params.probe_row_size
            + params.probe_batch_size
            * params.avg_matches
            * params.output_row_size
            * (1.0 + self.c_merge_ratio)
        )

    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[JoinParams]:
        if op_context is None or "right_chars" not in op_context:
            raise ValueError(
                "JoinMemoryModel.solve: op_context['right_chars'] required"
            )
        right_chars: DataChars = op_context["right_chars"]
        how: str = op_context.get("how", "inner")
        if how not in ("inner", "left"):
            raise ValueError(
                f"JoinMemoryModel.solve: how must be 'inner'|'left', got {how!r}"
            )
        left_chars = data_chars

        # P: max от shuffle для обеих сторон
        left_shuffle = self.shuffle_model.solve(
            available_bytes, left_chars, safety=safety
        )
        right_shuffle = self.shuffle_model.solve(
            available_bytes, right_chars, safety=safety
        )
        if left_shuffle is None or right_shuffle is None:
            return None
        P = max(left_shuffle.num_partitions, right_shuffle.num_partitions)
        if P < 2:
            return None

        # build_side
        if how == "left":
            build_side = "right"  # left preserved → right = build
        else:  # inner
            left_size = (left_chars.k_est or left_chars.n_rows) * left_chars.row_size_bytes
            right_size = (right_chars.k_est or right_chars.n_rows) * right_chars.row_size_bytes
            build_side = "left" if left_size <= right_size else "right"

        build_chars = left_chars if build_side == "left" else right_chars
        probe_chars = right_chars if build_side == "left" else left_chars

        budget = int(available_bytes * safety)
        build_budget = int(budget * self.build_ratio)
        probe_budget = int(budget * self.probe_ratio)
        output_budget = int(budget * self.output_ratio)

        # build_partition per partition
        build_k = build_chars.k_est or build_chars.n_rows
        build_partition_bytes = (build_k / P) * build_chars.row_size_bytes
        if build_partition_bytes > build_budget:
            return None

        # avg_matches (upper bound)
        k_left = left_chars.k_est or left_chars.n_rows
        k_right = right_chars.k_est or right_chars.n_rows
        if k_left > 0 and k_right > 0 and left_chars.n_rows > 0:
            output_upper = (
                left_chars.n_rows * right_chars.n_rows / max(k_left, k_right)
            )
            avg_matches = max(1.0, output_upper / left_chars.n_rows)
        else:
            avg_matches = 1.0

        # row_size_output: консервативно = probe_row + build_row
        # (реально = probe_rest + build_rest + on_keys, но probe_row и
        # build_row включают on-keys, что даёт верхнюю оценку).
        row_size_out = probe_chars.row_size_bytes + build_chars.row_size_bytes

        # probe_batch: из двух ограничений
        probe_batch_by_probe = max(
            1, probe_budget // max(1, probe_chars.row_size_bytes)
        )
        denom = avg_matches * row_size_out * (1.0 + self.c_merge_ratio)
        probe_batch_by_output = max(1, int(output_budget / max(1.0, denom)))
        probe_batch_size = min(probe_batch_by_probe, probe_batch_by_output)

        # max_output_rows: картезиан-стопор (upper bound actual output).
        # worst-case pd.merge output = |build| × |probe| строк.
        # Допускаем не более max_output_rows при пике output_budget.
        max_output_rows = max(
            1, int(output_budget / max(1.0, row_size_out * (1.0 + self.c_merge_ratio)))
        )

        params = JoinParams(
            build_side=build_side,
            how=how,
            num_partitions=int(P),
            probe_batch_size=int(probe_batch_size),
            build_partition_bytes=int(build_partition_bytes),
            max_output_rows=int(max_output_rows),
            right_chars=right_chars,
            probe_row_size=int(probe_chars.row_size_bytes),
            output_row_size=int(row_size_out),
            avg_matches=float(avg_matches),
        )

        # Инвариант: estimate <= budget. Консервативные округления могут
        # дать крошечное превышение; уменьшаем probe_batch_size.
        if self.estimate(params, left_chars) > budget:
            params.probe_batch_size -= 1
            if params.probe_batch_size < 1:
                return None
            if self.estimate(params, left_chars) > budget:
                return None
        return params

    def describe(self) -> str:
        return (
            "JoinMemoryModel: "
            "f = build_partition + probe_batch + output_batch*(1+c_merge); "
            f"build_ratio={self.build_ratio:.2f}, "
            f"probe_ratio={self.probe_ratio:.2f}, "
            f"output_ratio={self.output_ratio:.2f}, "
            f"c_merge={self.c_merge_ratio:.2f}"
        )


register_model(JoinMemoryModel())
