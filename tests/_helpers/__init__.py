

def _row_count(obj):
    """Row count for pa.Table | CadwDataFrame (v0.9.95).

    pa.Table has .num_rows; CadwDataFrame has .count().
    """
    if hasattr(obj, "count") and callable(getattr(obj, "count", None)):
        return obj.count()
    if hasattr(obj, "num_rows"):
        return obj.num_rows
    raise TypeError(
        f"_row_count: unsupported type {type(obj).__name__}")
