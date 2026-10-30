"""Regenerate Xenium figures from the downloadable data bundles."""
import argparse
from pathlib import Path
import shutil
import tempfile

from analyses.figures.common import figure_dir, write_index
from analyses.figures.scripts.xenium.build import build as figure_build
from analyses.figures.scripts.xenium.map_data import evidence_root
from .coherence import check, check_figure


def publish(replace=False):
    destinations = [figure_dir(name) for name in ('FigureS7', 'FigureS8')]
    if not replace and any(path.exists() for path in destinations):
        raise ValueError('Figures exist; use --replace-generated')
    destinations[0].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.xenium-figures-', dir=destinations[0].parent) as directory:
        stage = Path(directory) / 'FigureS7'
        figure_build(stage, evidence_root())
        audit = check(evidence_root(), stage)
        audit.update(check_figure(stage, evidence_root()))
        for dest in destinations:
            # Only generated files are replaced; manually assembled layouts remain untouched.
            shutil.copytree(Path(directory) / dest.name, dest, dirs_exist_ok=True)
    write_index()
    print('Updated Figures S7 and S8; input and supplementary bundles unchanged')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-generated', action='store_true')
    args = parser.parse_args()
    publish(args.replace_generated)


if __name__ == '__main__':
    main()
