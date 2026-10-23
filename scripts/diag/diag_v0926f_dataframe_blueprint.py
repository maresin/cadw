#!/usr/bin/env python3
# v0.9.26f-diagnostic: dataframe.py blueprint for B.1 9-module split.
# READ-ONLY. No mutation, no commit.
# ASCII-only. Discovery-first (section 13.4/288).

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


DF_PATH = "cadw/dataframe/_core.py"


# ---------- 9 module categories ----------
# Key: proposed module name. Value: heuristics for matching
# method/property names, docstring keywords, and public symbols.

MODULE_HEURISTICS = {
    "_core": {
        "names": (
            "__init__", "__repr__", "__repr_html__", "__len__",
            "columns", "schema", "shape", "dtypes",
            "_spill_dir", "_new_dataset_dir", "_wrap_result_table",
            "_memory_guard", "_dataset", "_columns", "_row_filters",
            "_index_kind", "_index_start",
        ),
        "desc": "Construction, properties, repr, internal state access",
    },
    "_columns": {
        "names": (
            "__getitem__", "_ColumnProxy", "ColumnProxy",
            "select_dtypes", "dtypes",
        ),
        "desc": "Column access, proxies, dtype inspection",
    },
    "_filter": {
        "names": (
            "PredicateFilter", "MaskFilter", "where", "filter",
        ),
        "desc": "Row filters, mask application, where",
    },
    "_indexing": {
        "names": (
            "iloc", "loc", "at", "iat",
            "set_index", "reset_index", "reindex",
            "sort_values", "sort_index", "head", "tail",
        ),
        "desc": "Indexing, slicing, sort, reindex",
    },
    "_aggregations": {
        "names": (
            "groupby", "agg", "describe", "value_counts",
            "corr", "cov", "nunique", "isna", "isna_sum",
            "count", "sum", "mean", "std", "var", "min", "max",
            "median", "quantile", "nlargest", "nsmallest",
            "mode",
        ),
        "desc": "Groupby, describe, statistics",
    },
    "_transformers": {
        "names": (
            "astype", "drop", "rename", "dropna", "fillna",
            "ffill", "bfill",
            "shift", "diff", "pct_change",
            "cut", "qcut", "duplicated", "drop_duplicates",
            "apply", "explode",
        ),
        "desc": "Transforms: astype, drop, fill, shift, cut, apply",
    },
    "_ops": {
        "names": (
            "isin", "between", "idxmax", "idxmin",
        ),
        "desc": "Predicate + reduction ops (isin/between/idx*)",
    },
    "_exports": {
        "names": (
            "to_batches", "to_parquet", "to_pandas",
            "write_parquet", "write_batches",
            "from_parquet", "from_table", "from_batches",
        ),
        "desc": "Serialization, IO",
    },
    "_compat": {
        "names": (
            "pandas_compat", "_to_pandas_safe",
        ),
        "desc": "Pandas interop, dtype coercion",
    },
}


def indent(s, n):
    pad = " " * n
    return "\n".join(pad + ln for ln in s.splitlines())


# ---------- 1. AST inventory ----------

def ast_scan():
    src = Path(DF_PATH).read_text()
    tree = ast.parse(src)
    lines = src.split("\n")

    # Top-level classes
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            end = max(
                getattr(n, "end_lineno", node.lineno)
                for n in ast.walk(node)
            )
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_end = max(
                        getattr(n, "end_lineno", item.lineno)
                        for n in ast.walk(item)
                    )
                    methods.append({
                        "name": item.name,
                        "lineno": item.lineno,
                        "end": m_end,
                        "loc": m_end - item.lineno + 1,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                    })
            classes[node.name] = {
                "lineno": node.lineno,
                "end": end,
                "loc": end - node.lineno + 1,
                "methods": methods,
            }

    # Top-level imports
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "kind": "import",
                    "name": alias.name,
                    "asname": alias.asname,
                    "lineno": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    "kind": "from",
                    "module": node.module,
                    "name": alias.name,
                    "asname": alias.asname,
                    "lineno": node.lineno,
                })

    return {
        "src": src,
        "src_lines": len(lines),
        "tree": tree,
        "classes": classes,
        "imports": imports,
    }


# ---------- 2. print actual structure ----------

def print_structure(info):
    print("\n" + "=" * 72)
    print("ACTUAL FILE STRUCTURE (" + DF_PATH + ")")
    print("=" * 72)
    print("\ntotal lines: " + str(info["src_lines"]))
    print("imports: " + str(len(info["imports"])))
    print("top-level classes: " + str(len(info["classes"])))

    print("\n--- imports (first 30) ---")
    for imp in info["imports"][:30]:
        if imp["kind"] == "import":
            asn = " as " + imp["asname"] if imp["asname"] else ""
            print("  line " + str(imp["lineno"]).rjust(4)
                  + "| import " + imp["name"] + asn)
        else:
            asn = " as " + imp["asname"] if imp["asname"] else ""
            print("  line " + str(imp["lineno"]).rjust(4)
                  + "| from " + str(imp["module"]) + " import "
                  + imp["name"] + asn)
    if len(info["imports"]) > 30:
        print("  ... (" + str(len(info["imports"]) - 30) + " more)")

    print("\n--- classes ---")
    for cname, cinfo in info["classes"].items():
        print("  " + cname + ": lines " + str(cinfo["lineno"])
              + "-" + str(cinfo["end"])
              + " (" + str(cinfo["loc"]) + " LOC, "
              + str(len(cinfo["methods"])) + " methods)")


# ---------- 3. method inventory per class ----------

def method_table(info):
    print("\n" + "=" * 72)
    print("METHOD INVENTORY")
    print("=" * 72)
    for cname, cinfo in info["classes"].items():
        print("\n--- " + cname + " (" + str(len(cinfo["methods"]))
              + " methods) ---")
        for m in sorted(cinfo["methods"], key=lambda x: -x["loc"])[:100]:
            prefix = "async " if m["is_async"] else ""
            print("  line " + str(m["lineno"]).rjust(5)
                  + " " + str(m["loc"]).rjust(4) + " LOC  "
                  + prefix + m["name"])
        if len(cinfo["methods"]) > 100:
            print("  ... (" + str(len(cinfo["methods"]) - 100)
                  + " more)")


# ---------- 4. module assignment ----------

def assign_modules(info):
    print("\n" + "=" * 72)
    print("MODULE ASSIGNMENT (9 categories)")
    print("=" * 72)

    # Build name -> module map from heuristics
    name_to_mod = {}
    for mod, spec in MODULE_HEURISTICS.items():
        for nm in spec["names"]:
            name_to_mod[nm] = mod

    assignment = {mod: [] for mod in MODULE_HEURISTICS}
    assignment["_core_UNMATCHED"] = []  # fallback bucket

    total_loc = 0
    total_methods = 0
    matched_loc = 0

    for cname, cinfo in info["classes"].items():
        for m in cinfo["methods"]:
            total_loc += m["loc"]
            total_methods += 1
            mod = name_to_mod.get(m["name"])
            if mod is None:
                mod = "_core_UNMATCHED"
            assignment[mod].append({
                "class": cname,
                "name": m["name"],
                "loc": m["loc"],
                "lineno": m["lineno"],
            })
            matched_loc += m["loc"] if mod != "_core_UNMATCHED" else 0

    for mod in MODULE_HEURISTICS:
        items = assignment[mod]
        if not items:
            print("\n[" + mod + "] (empty - heuristic matched nothing)")
            continue
        loc = sum(i["loc"] for i in items)
        print("\n[" + mod + "]  "
              + str(len(items)) + " methods, "
              + str(loc) + " LOC  - " + MODULE_HEURISTICS[mod]["desc"])
        for it in sorted(items, key=lambda x: -x["loc"]):
            print("  " + str(it["loc"]).rjust(4) + " L  "
                  + it["class"] + "." + it["name"]
                  + "  (line " + str(it["lineno"]) + ")")

    unmatched = assignment["_core_UNMATCHED"]
    if unmatched:
        print("\n[DRIFT: _core_UNMATCHED]  "
              + str(len(unmatched)) + " methods, "
              + str(sum(i["loc"] for i in unmatched)) + " LOC")
        print("  (not in any heuristic - extend MODULE_HEURISTICS "
              "or accept into _core)")
        for it in sorted(unmatched, key=lambda x: -x["loc"]):
            print("  " + str(it["loc"]).rjust(4) + " L  "
                  + it["class"] + "." + it["name"]
                  + "  (line " + str(it["lineno"]) + ")")

    match_pct = 100.0 * matched_loc / max(total_loc, 1)
    print("\n--- coverage ---")
    print("total methods: " + str(total_methods))
    print("total LOC in methods: " + str(total_loc))
    print("matched LOC: " + str(matched_loc)
          + " (" + ("%.1f" % match_pct) + "%)")
    print("unmatched LOC: " + str(total_loc - matched_loc)
          + " (" + ("%.1f" % (100 - match_pct)) + "%)")


# ---------- 5. AST-guard test dependencies ----------

def grep_tests():
    print("\n" + "=" * 72)
    print("AST-GUARD TEST DEPENDENCIES")
    print("=" * 72)
    print("\ntests that reference cadw/dataframe.py source path")
    print("(would need path updates after split)")

    hits = []
    for p in Path("tests").rglob("*.py"):
        src = p.read_text()
        n = src.count("cadw/dataframe/_core.py")
        if n:
            hits.append((str(p), n))

    if not hits:
        print("  (none found - good)")
    else:
        for fname, n in sorted(hits):
            print("  " + fname + ": " + str(n) + " reference(s)")

    print("\ntests using assert_no_call_in_source on CadwDataFrame")
    hits = []
    for p in Path("tests").rglob("*.py"):
        src = p.read_text()
        if "assert_no_call_in_source" in src and (
            "CadwDataFrame" in src or "dataframe" in src
        ):
            hits.append(str(p))
    if not hits:
        print("  (none found)")
    else:
        for fname in sorted(hits):
            print("  " + fname)

    print("\npublic API re-export checks")
    init_path = Path("cadw/__init__.py")
    if init_path.exists():
        src = init_path.read_text()
        if "CadwDataFrame" in src:
            for i, ln in enumerate(src.splitlines(), 1):
                if "CadwDataFrame" in ln:
                    print("  cadw/__init__.py:" + str(i)
                          + "  " + ln.strip()[:100])


# ---------- 6. dependency graph (rough) ----------

def dependency_signals(info):
    print("\n" + "=" * 72)
    print("DEPENDENCY SIGNALS (for migration order)")
    print("=" * 72)
    print("\nFor each method, count internal self.X references to:")
    print("  - self._dataset, self._memory_guard, self._columns,")
    print("    self._row_filters, self._spill_dir")
    print("  (HIGH self.X => core-dependent; LOW self.X => leaf, "
          "extract first)")

    leaf_candidates = []
    core_dependent = []
    for cname, cinfo in info["classes"].items():
        src = info["src"]
        lines = src.split("\n")
        for m in cinfo["methods"]:
            snippet = "\n".join(
                lines[m["lineno"] - 1:m["end"]]
            )
            hits = sum(
                snippet.count("self." + attr)
                for attr in (
                    "_dataset", "_memory_guard", "_columns",
                    "_row_filters", "_spill_dir", "_index_kind",
                    "_index_start", "_iter_filtered_batches",
                    "_new_dataset_dir", "_wrap_result_table",
                )
            )
            if hits <= 1 and m["loc"] >= 3:
                leaf_candidates.append((cname, m["name"],
                                       m["loc"], hits))
            elif hits >= 8:
                core_dependent.append((cname, m["name"],
                                       m["loc"], hits))

    print("\n--- leaf candidates (low self.X, extract early) ---")
    for cname, mname, loc, hits in sorted(
        leaf_candidates, key=lambda x: x[3]
    )[:15]:
        print("  " + cname + "." + mname
              + "  (" + str(loc) + "L, " + str(hits) + " self.X)")

    print("\n--- core-dependent (high self.X, stay in _core) ---")
    for cname, mname, loc, hits in sorted(
        core_dependent, key=lambda x: -x[3]
    )[:15]:
        print("  " + cname + "." + mname
              + "  (" + str(loc) + "L, " + str(hits) + " self.X)")


# ---------- 7. recommended migration order + rollback ----------

def migration_plan():
    print("\n" + "=" * 72)
    print("RECOMMENDED MIGRATION ORDER + ROLLBACK")
    print("=" * 72)
    print("""
Standard order for a large-class split (leaf-first, core-last):

  Step 1: _exports   -- to_batches, to_parquet, to_pandas,
                        from_*; near-zero self.X; isolated.
  Step 2: _compat    -- pandas interop shims; independent.
  Step 3: _ops       -- isin/between/idxmax/idxmin; small, well-tested
                        in Phase C (v0.9.25g/h).
  Step 4: _filter    -- PredicateFilter/MaskFilter; used by many but
                        self-contained.
  Step 5: _indexing  -- iloc/loc/at/iat/sort/reindex; depends on _core.
  Step 6: _aggregations -- groupby/describe/corr; depends on _core +
                           ops primitives.
  Step 7: _transformers -- astype/drop/fill/shift/cut/apply; uses
                           _core heavily.
  Step 8: _columns   -- __getitem__/_ColumnProxy; central glue.
  Step 9: _core      -- stays as `dataframe/__init__.py` or
                        `dataframe/_core.py`; CadwDataFrame base
                        remains.

  CadwDataFrame class: use MIXIN composition --
    class CadwDataFrame(_ExportsMixin, _CompatMixin, _OpsMixin,
                        _FilterMixin, _IndexingMixin,
                        _AggregationsMixin, _TransformersMixin,
                        _ColumnsMixin, _CoreBase):
        pass
  Each mixin: methods moved verbatim, no logic change.

Rollback per step:
  - Each extraction = 1 patch (v0.9.26g, h, i, ...).
  - If fast suite fails: `git revert <sha>` + `git checkout
    cadw/dataframe.py` (git holds prior version). No permanent
    state change beyond the failed commit.
  - If AST-guard test fails (path reference): update test path
    as part of the SAME patch, not a separate one.

Verification per step:
  - Full fast suite (2488 tests) green.
  - `import cadw; CadwDataFrame` still exports CadwDataFrame.
  - Whitelist (34 keys) unchanged.

Discovery gates for extraction Step N:
  - Diff-size sanity: moved LOC within +/-5% of estimate.
  - Public surface unchanged: `dir(CadwDataFrame)` superset of
    prior (no removed methods).
  - Method identity: `CadwDataFrame.method is Prior.method`
    (byte-equal __code__) -- optional but strong.
""")


# ---------- 8. main ----------

def main():
    print("=== v0.9.26f-diagnostic: dataframe.py blueprint ===")

    p = Path(DF_PATH)
    if not p.exists():
        raise SystemExit("missing " + DF_PATH)

    info = ast_scan()
    print_structure(info)
    method_table(info)
    assign_modules(info)
    grep_tests()
    dependency_signals(info)
    migration_plan()

    print("\n" + "=" * 72)
    print("NEXT STEP DECISION TREE")
    print("=" * 72)
    print("""
Read the coverage section above.

  match_pct >= 70%      -- blueprint solid; write v0.9.26g-split-extract-1
                           (Step 1: _exports) immediately.
  match_pct 50-69%      -- extend MODULE_HEURISTICS in this diag,
                           re-run until >= 70%.
  match_pct < 50%       -- file structure very different from
                           expected; re-inspect AST output carefully.

  DRIFT bucket          -- if >10 unmatched methods, name them and
  (_core_UNMATCHED)        decide: _core (default) or extend a
                           heuristic. Do not force-place.

  AST-guard hits        -- note which tests reference the source
                           path; those paths are updated in the SAME
                           extraction patch that moves the code
                           (not separate).

  leaf candidates       -- the extraction order recommendation is
                           validated if Step 1 (_exports) methods
                           show up here. If not, re-order.
""")


if __name__ == "__main__":
    main()