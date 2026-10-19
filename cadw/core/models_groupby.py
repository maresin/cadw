"""
GroupByMemoryModel (§9.2.4).

Композиция: HashShuffle + _local_groupby + concat.

    f_groupby = max(f_shuffle, f_local_groupby)
    f_local_groupby = partition_bytes + dict_bytes + c_temp_ratio * dict_bytes
    partition_bytes = (K_est / P) * row_size
    dict_bytes = (K_est / P) * size_of_key_entry

recurse_depth: upper bound на число уровней split внутри партиции.
  Если K_est/P > max_dict_keys — split по P_recurse до тех пор, пока
  K_per_partition' <= max_dict_keys и partition_bytes' <= file_budget.
  depth = max(d_dict, d_file); None, если > MAX_DEPTH.

Параметры partition_ratio, dict_ratio, c_temp_ratio, p_recurse, max_depth —
CALIBRATION DEFAULT (11-A-4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from cadw.core.models import (
    DataChars,
    OpMemoryModel,
    Params,
    get_model,
    register_model,
)
from cadw.core.models_shuffle import HashShuffleMemoryModel, HashShuffleParams


@dataclass
class GroupByParams(Params):
    shuffle: HashShuffleParams
    max_dict_keys: int
    recurse_depth: int
    p_recurse: int


def _depth_for(needed: float, capacity: float, p_recurse: int) -> int:
    if capacity <= 0:
        return 10**6  # effectively infinite
    if needed <= capacity:
        return 0
    return int(math.ceil(math.log(needed / capacity) / math.log(p_recurse)))


class GroupByMemoryModel(OpMemoryModel):
    op_name = "groupby"

    def __init__(
        self,
        shuffle_model: Optional[HashShuffleMemoryModel] = None,
        *,
        p_recurse: int = 4,             # CALIBRATION DEFAULT
        max_depth: int = 5,             # CALIBRATION DEFAULT
        partition_ratio: float = 0.35,  # доля budget под partition_bytes
        dict_ratio: float = 0.25,       # доля budget под dict_bytes
        c_temp_ratio: float = 1.0,      # temp ≈ dict_bytes
    ) -> None:
        self.shuffle_model = shuffle_model or get_model("hash_shuffle")
        if not isinstance(self.shuffle_model, HashShuffleMemoryModel):
            raise TypeError("shuffle_model must be HashShuffleMemoryModel")
        if p_recurse < 2:
            raise ValueError("p_recurse must be >= 2")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        for name, r in (
            ("partition_ratio", partition_ratio),
            ("dict_ratio", dict_ratio),
            ("c_temp_ratio", c_temp_ratio),
        ):
            if r < 0:
                raise ValueError(f"{name} must be >= 0")
        if partition_ratio + dict_ratio * (1 + c_temp_ratio) > 1.0:
            raise ValueError(
                "partition_ratio + dict_ratio*(1+c_temp_ratio) must be <= 1.0; "
                f"got {partition_ratio} + {dict_ratio}*(1+{c_temp_ratio})"
            )
        self.p_recurse = int(p_recurse)
        self.max_depth = int(max_depth)
        self.partition_ratio = float(partition_ratio)
        self.dict_ratio = float(dict_ratio)
        self.c_temp_ratio = float(c_temp_ratio)

    def estimate(self, params: GroupByParams, data_chars: DataChars) -> int:
        if not isinstance(params, GroupByParams):
            raise TypeError(f"expected GroupByParams, got {type(params).__name__}")
        if data_chars.k_est is None or data_chars.size_of_key_entry_bytes is None:
            raise ValueError(
                "GroupByMemoryModel.estimate: "
                "k_est and size_of_key_entry_bytes required"
            )

        f_shuffle = self.shuffle_model.estimate(params.shuffle, data_chars)

        # Estimate — для листа дерева рекурсии. После `recurse_depth` split'ов
        # каждая подпартиция содержит K/P/p_recurse^d ключей. Родительская
        # партиция при split не материализуется целиком: hash_shuffle
        # обрабатывает батчами, RAM ≤ f_shuffle (на подпартиции ≤ f_shuffle
        # для полного датасета, т.к. подпартиция меньше). Поэтому лист
        # доминирует в пике локальной агрегации.
        p_recurse = params.p_recurse if params.recurse_depth > 0 else 1
        divisor = p_recurse ** params.recurse_depth
        k_per_leaf = data_chars.k_est / params.shuffle.num_partitions / divisor
        partition_leaf = k_per_leaf * data_chars.row_size_bytes
        dict_leaf = k_per_leaf * data_chars.size_of_key_entry_bytes
        f_local = partition_leaf + dict_leaf + self.c_temp_ratio * dict_leaf

        return int(max(f_shuffle, f_local))

    def solve(
        self,
        available_bytes: int,
        data_chars: DataChars,
        *,
        safety: float = 0.7,
        op_context: Optional[dict] = None,
    ) -> Optional[GroupByParams]:
        if data_chars.k_est is None:
            raise ValueError("GroupByMemoryModel.solve: k_est required")
        if data_chars.size_of_key_entry_bytes is None:
            raise ValueError(
                "GroupByMemoryModel.solve: size_of_key_entry_bytes required"
            )

        shuffle_params = self.shuffle_model.solve(
            available_bytes, data_chars, safety=safety
        )
        if shuffle_params is None:
            return None

        budget = int(available_bytes * safety)
        max_dict_keys = max(
            1, int(budget * self.dict_ratio / data_chars.size_of_key_entry_bytes)
        )
        file_budget = budget * self.partition_ratio
        if file_budget <= 0:
            return None

        k_per = data_chars.k_est / shuffle_params.num_partitions
        if k_per <= 0:
            # Пустой dataset
            return GroupByParams(
                shuffle=shuffle_params,
                max_dict_keys=max_dict_keys,
                recurse_depth=0,
                p_recurse=self.p_recurse,
            )

        d_dict = _depth_for(k_per, max_dict_keys, self.p_recurse)
        file_bytes = k_per * data_chars.row_size_bytes
        d_file = _depth_for(file_bytes, file_budget, self.p_recurse)
        recurse_depth = max(d_dict, d_file)
        if recurse_depth > self.max_depth:
            return None

        params = GroupByParams(
            shuffle=shuffle_params,
            max_dict_keys=int(max_dict_keys),
            recurse_depth=int(recurse_depth),
            p_recurse=self.p_recurse,
        )
        if self.estimate(params, data_chars) > budget:
            return None
        return params

    def describe(self) -> str:
        return (
            "GroupByMemoryModel: f = max(f_shuffle, f_local_groupby); "
            "recurse_depth = ceil(log((K/P)/max_dict_keys)/log(p_recurse)) "
            "(partition_ratio=%.2f, dict_ratio=%.2f, p_recurse=%d, max_depth=%d)"
            % (
                self.partition_ratio,
                self.dict_ratio,
                self.p_recurse,
                self.max_depth,
            )
        )


register_model(GroupByMemoryModel())
