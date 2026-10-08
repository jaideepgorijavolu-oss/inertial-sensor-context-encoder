"""Re-derive every ssl_knn selection and test prediction with the fixed knn_predict; report differences."""
import json
import os

import numpy as np
import torch

from src.lowlabel import embed, encoder_from, fit_knn, knn_predict
from src.lowlabel_data import StudyData, load_test
from src.ssl import standardize

R = "results/sensor_contrastive"
dev = torch.device("cuda")  # GPU, as the study ran
Xte, yte, _ = load_test("data/UCI HAR Dataset")
out = {}
for run, bl in (("mvs", 4), ("budgets_5_25", 4), ("iid", 1)):
    data = StudyData("data/UCI HAR Dataset", block_len=bl)
    mean, std = torch.tensor(data.scaler.mean.reshape(-1)), torch.tensor(data.scaler.std.reshape(-1))
    norm = lambda X: standardize(torch.tensor(X, device=dev), mean.to(dev), std.to(dev))
    sel = json.load(open(f"{R}/{run}/selections.json"))["entries"]
    test = json.load(open(f"{R}/{run}/test_metrics.json"))["entries"]
    for key, e in sel.items():
        if e["condition"] != "ssl_knn":
            continue
        ck = torch.load(os.path.join(R, run, e["checkpoint"]), map_location=dev, weights_only=False)
        enc = encoder_from(ck["encoder"], dev)
        sub = data.subsets(e["seed"], e["budget"])
        assert list(sub["train"]) == list(ck["ref_ids"])
        F_tr, F_va = embed(enc, norm(data.X_raw[sub["train"]])), embed(enc, norm(data.X_raw[sub["val"]]))
        new = fit_knn(F_tr, data.y[sub["train"]], F_va, data.y[sub["val"]])
        same_sel = (new["k"], new["weighted"]) == (e["hparams"]["k"], e["hparams"]["weighted"])
        pred = knn_predict(F_tr, data.y[sub["train"]], embed(enc, norm(Xte)), e["hparams"]["k"], e["hparams"]["weighted"])
        n_diff = int((pred != np.array(test[key]["predictions"])).sum())
        out[f"{run}/{key}"] = {"selection_unchanged": same_sel, "test_predictions_changed": n_diff,
                               "weighted": e["hparams"]["weighted"], "k": e["hparams"]["k"]}
        print(run, key, out[f"{run}/{key}"], flush=True)
json.dump(out, open(f"{R}/knn_fix_recheck.json", "w"), indent=2)
print("all unchanged:", all(v["selection_unchanged"] and v["test_predictions_changed"] == 0 for v in out.values()))
