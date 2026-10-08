"""
Replay every selected checkpoint on the test split and compare with the published predictions.

    PYTHONPATH=. python scripts/replay_checkpoints.py [--cpu]

Each run's selections and checkpoints are copied into a temporary directory and evaluated with the
current `evaluate-test` code, so the published run directories are never written to.
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from src import lowlabel

R = os.path.join("results", "sensor_contrastive")
RUNS = ("mvs", "budgets_5_25", "iid", "ablation_norot")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--data-dir", default=lowlabel.DEFAULT_DATA_DIR)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    report = {"device": str(device), "runs": {}}
    for run in RUNS:
        src = os.path.join(R, run)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(os.path.join(src, "selections.json"), tmp)
            shutil.copytree(os.path.join(src, "models"), os.path.join(tmp, "models"))
            args = argparse.Namespace(run_dir=tmp, data_dir=a.data_dir)
            new = lowlabel.evaluate_test(args, device, log=lambda *_: None)["entries"]
        old = json.load(open(os.path.join(src, "test_metrics.json")))["entries"]
        diffs = {k: int((np.array(new[k]["predictions"]) != np.array(old[k]["predictions"])).sum()) for k in old}
        report["runs"][run] = {"models": len(diffs), "models_with_changed_predictions": sum(d > 0 for d in diffs.values()),
                               "changed_predictions": {k: d for k, d in diffs.items() if d}}
        print(run, report["runs"][run], flush=True)
    json.dump(report, open(os.path.join(R, f"replay_{device.type}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
