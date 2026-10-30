"""Mouse inputs and numerical evidence; no filesystem writes on import."""
import os
from pathlib import Path
from tilescope_results.paths import data_root


def locations(output_root=None):
    data = Path(os.environ.get("TILESCOPE_DATA_MOUSE_ROOT", str(data_root() / "mouse")))
    analysis = Path(__file__).resolve().parent
    output = Path(output_root) if output_root else data / "mouse_gbm_spatial_programs" / "phase6_harmonized" / "evidence"
    return data, analysis, output
