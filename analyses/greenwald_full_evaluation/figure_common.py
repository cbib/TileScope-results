"""Shared audited input handling and plotting utilities; no scientific selection."""
from __future__ import annotations
from tilescope_results.paths import data_root
import json
import os
import sys
from pathlib import Path
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tilescope_figure_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry.polygon import orient
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

RUN_STATUS = data_root() / "heiland_tyrosh/CurrentResults/run_accounting/run_status.csv"
EXPECTED_CORE = "ef8ab0df1332bda822a2cf3240fc856c466ed534"


def one(frame, **selectors):
    mask = pd.Series(True, index=frame.index)
    for key, value in selectors.items():
        mask &= frame[key].astype(str).eq(str(value))
    result = frame.loc[mask]
    if len(result) != 1:
        raise ValueError(f"Expected one row for {selectors}, found {len(result)}")
    return result.iloc[0]


class Inputs:
    def __init__(self):
        self.paths = set()
        self._run_status = None

    def csv(self, path):
        self.paths.add(Path(path).resolve())
        return pd.read_csv(path, dtype={"roi_id": str, "component_id": str})

    def json(self, path):
        self.paths.add(Path(path).resolve())
        return json.loads(Path(path).read_text())

    def run_status(self):
        if self._run_status is None:
            status = self.csv(RUN_STATUS)
            required = {"sample_id", "pair", "status", "core_sha"}
            if not required.issubset(status) or status.duplicated(["sample_id", "pair"]).any():
                raise ValueError("Invalid Greenwald run inventory")
            if not status.status.eq("complete").all() or not status.core_sha.eq(EXPECTED_CORE).all():
                raise ValueError("Run inventory is not complete at the expected core revision")
            self._run_status = status
        return self._run_status

    def run(self, results, sample, pair):
        folder = results / sample / pair
        row = self.run_status()
        row = row[row.sample_id.eq(sample) & row.pair.eq(pair)]
        if len(row) != 1:
            raise ValueError(f"Run is absent from the completed inventory: {sample}/{pair}")
        meta = self.json(folder / "run_metadata.json")
        config = self.json(folder / "effective_config.json")
        if meta.get("core_sha") not in (None, EXPECTED_CORE):
            raise ValueError(f"Unexpected embedded core SHA in {folder}")
        if config["run_config"]["tessellation"]["n"] != 10:
            raise ValueError(f"Expected n=10 in {folder}")
        return folder, meta


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def map_base(ax, frame, mask, crop=False):
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError("Empty map selection")
    extent = selected if crop else frame
    xmin, xmax = extent.x.min(), extent.x.max(); ymin, ymax = extent.y.min(), extent.y.max()
    pad = max(xmax-xmin, ymax-ymin) * 0.07
    visible = frame.x.between(xmin-pad, xmax+pad) & frame.y.between(ymin-pad, ymax+pad)
    ax.scatter(frame.loc[visible & ~mask, "x"], frame.loc[visible & ~mask, "y"],
               s=3, c="#DFE3E8", linewidths=0, rasterized=True)
    ax.set(xlim=(xmin-pad, xmax+pad), ylim=(ymin-pad, ymax+pad), aspect="equal")
    ax.set_axis_off()


def expression_map(ax, frame, mask, column, label, crop=False):
    map_base(ax, frame, mask, crop)
    values = frame[column].to_numpy(float)
    limit = max(float(np.quantile(np.abs(values), 0.98)), 1e-12)
    sc = ax.scatter(frame.loc[mask, "x"], frame.loc[mask, "y"],
                    c=frame.loc[mask, column], cmap="RdBu_r", vmin=-limit, vmax=limit,
                    s=(12 if int(np.sum(mask)) < 100 else 5) if crop else 5,
                    linewidths=0, rasterized=True)
    ax.set_title(label, pad=5)
    ax.figure.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.035, pad=0.025,
                       ticks=[-limit, 0, limit], format="%.2g")
    return limit


def component_mask(frame, ledger, component):
    if ledger.spot_index.duplicated().any() or set(ledger.spot_index) != set(range(len(frame))):
        raise ValueError("Ledger indices must identify every input row exactly once")
    ledger = ledger.sort_values("spot_index")
    if not np.array_equal(frame.roi_id.astype(str), ledger.roi_id.astype(str)):
        raise ValueError("Input ROI order differs from source ledger")
    return ledger.component_id.eq(component).fillna(False).to_numpy()


def save(fig, stem, output, pdf):
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


def polygon_patch(poly):
    """Preserve native polygon holes; never reconstruct support from centroids."""
    poly=orient(poly,sign=1.)
    paths=[]
    for ring in [poly.exterior,*poly.interiors]:
        xy=np.asarray(ring.coords)
        paths.append(MplPath(xy,[MplPath.MOVETO]+[MplPath.LINETO]*(len(xy)-2)+[MplPath.CLOSEPOLY]))
    return PathPatch(MplPath.make_compound_path(*paths))
