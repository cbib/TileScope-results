"""Frozen Meylan cohort identities and locations; no sample selection by results."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE_CELLS = {
    'P05_pre': 9433, 'P05_post': 15137,
    'P21_pre': 8348, 'P21_post': 175499,
    'P28_pre': 19471, 'P28_post': 97488,
    'P32_pre': 7729, 'P32_post': 112292,
    'P34_pre': 61477, 'P34_post': 667500,
    'P45_pre': 19698, 'P45_post': 274787,
    'P46_pre': 51237, 'P46_post': 339858,
    'P51_pre': 6655, 'P51_post': 535337,
}
PAIR = 'Stemlike_Interferon'
DEFINITION_KEYS = ('pair', 'feature_a', 'feature_b', 'genes_a', 'genes_b', 'calculation')


def input_directory(root, sample):
    if sample not in SAMPLE_CELLS:
        raise ValueError(f'Unknown Meylan sample: {sample}')
    base = Path(root) / 'tilescope_inputs' / PAIR
    # Preserve the completed pilot's paths and bytes; new inputs are per sample.
    return base if sample == 'P28_post' else base / sample


def input_path(root, sample):
    return input_directory(root, sample) / f'{sample}_expression_data_{PAIR}.csv'


def same_definition(left, right):
    return all(left.get(k) == right.get(k) for k in DEFINITION_KEYS)
