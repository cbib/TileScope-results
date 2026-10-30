import numpy as np
import pandas as pd
import pytest

from analyses.xenium_feasibility.analyse import align_ledger, support_summary


def example():
    frame = pd.DataFrame({"roi_id": ["A", "A", "B"], "x": [0, 1, 2], "y": [0, 0, 0]})
    ledger = pd.DataFrame({"spot_index": [0, 1, 2], "roi_id": ["A", "A", "B"],
                           "tessellation_eligible": [True, np.nan, False],
                           "component_status": ["retained_component", "small_connected_component", "retained_component"],
                           "component_id": [0, np.nan, 1]})
    return frame, ledger


def test_null_eligibility_remains_in_denominator():
    frame, ledger = example()
    out = support_summary(frame, ledger).set_index("roi_id")
    assert out.loc["A", "eligible_fraction"] == .5
    assert out.loc["B", "eligible_fraction"] == 0
    assert out.n_cells.sum() == 3


def test_cell_identity_is_checked_before_summarizing():
    frame, ledger = example()
    ledger.loc[2, "spot_index"] = 1
    with pytest.raises(ValueError, match="uniquely indexed"):
        align_ledger(frame, ledger)


def test_order_can_be_restored_but_roi_drift_is_rejected():
    frame, ledger = example()
    assert align_ledger(frame, ledger.iloc[::-1]).spot_index.tolist() == [0, 1, 2]
    ledger.loc[0, "roi_id"] = "B"
    with pytest.raises(ValueError, match="ROI identities"):
        align_ledger(frame, ledger)
