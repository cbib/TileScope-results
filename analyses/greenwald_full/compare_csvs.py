"""Exact parsed-CSV comparison; a reported difference is not a scientific verdict."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compare(before: Path, after: Path):
    rows = []
    names = sorted({str(p.relative_to(root)) for root in (before, after)
                    for p in root.rglob('*') if p.name.endswith(('.csv', '.csv.gz'))})
    for name in names:
        left, right = before / name, after / name
        row = dict(file=name, status='identical', old_rows=None, new_rows=None,
                   changed_columns='', max_numeric_delta=0.0)
        if not left.is_file() or not right.is_file():
            row['status'] = 'added' if right.is_file() else 'removed'
            rows.append(row); continue
        a, b = pd.read_csv(left), pd.read_csv(right)
        row.update(old_rows=len(a), new_rows=len(b))
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            row['status'] = 'shape_or_schema_changed'
        else:
            changed = []
            for column in a:
                try:
                    pd.testing.assert_series_equal(a[column], b[column], check_exact=True)
                except AssertionError:
                    changed.append(column)
                    if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
                        av, bv = a[column].to_numpy(float), b[column].to_numpy(float)
                        good = np.isfinite(av) & np.isfinite(bv)
                        if good.any():
                            row['max_numeric_delta'] = max(row['max_numeric_delta'], float(np.max(np.abs(av[good] - bv[good]))))
            if changed:
                row['status'] = 'values_changed'
                row['changed_columns'] = ';'.join(changed)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.before.is_dir() or not args.after.is_dir():
        parser.error('Both input directories must exist')
    result = compare(args.before, args.after)
    if result.empty:
        parser.error('No CSVs found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.status.value_counts().to_string())
    raise SystemExit(0 if result.status.eq('identical').all() else 1)


if __name__ == '__main__':
    main()
