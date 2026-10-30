"""Integrity checks for the downloadable data bundles."""
from pathlib import Path
import hashlib

import pandas as pd


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(root):
    """Check the archive's relative file index, without following external paths."""
    root = Path(root)
    index = pd.read_csv(root / 'file_index.csv', keep_default_na=False)
    if index.file.duplicated().any():
        raise ValueError(f'Duplicate bundle entry: {root.name}')
    files = set()
    for row in index.itertuples(index=False):
        rel = Path(row.file)
        path = root / rel
        if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Nonlocal bundle entry: {row.file}')
        if path.stat().st_size != row.bytes or digest(path) != row.sha256:
            raise ValueError(f'Bundle file changed: {root.name}/{row.file}')
        files.add(path)
    return files


def main():
    """Verify downloaded files without modifying indexes or archives."""
    import argparse
    from tilescope_results.paths import input_data_root, supplementary_data_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', nargs='?', default='all',
                        choices=['all', 'inputs', '1', '2', '3', '4'])
    args = parser.parse_args()
    roots = [input_data_root()] if args.bundle in {'all', 'inputs'} else []
    roots += [supplementary_data_root(n) for n in range(1, 5)
              if args.bundle in {'all', str(n)}]
    for root in roots:
        print(f'{root.name}: {len(verify(root))} verified files')


if __name__ == '__main__':
    main()
