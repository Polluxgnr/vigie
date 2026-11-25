"""Train the supervised heads and the fusion, then evaluate everything on real thefts.

Data roles (no footage is used in two roles):
  * staged thefts (RetailS staged, 624 clips)      -> positives + hard negatives for the heads
  * RetailS holdout, first third of videos         -> easy negatives for the heads
  * RetailS holdout, second third                  -> normal reference for fusion and thresholds
  * RetailS holdout, last third                    -> false alarms per hour (never trained on)
  * RetailS real-world test (= PoseLift test)      -> final benchmark, real thefts
The normality model was trained on RetailS-fit, disjoint from all of the above.

Usage: python scripts/train_heads.py --normality runs/normality_retails_fit/model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from vigie.data import get_split
from vigie.eval import (AlertPolicy, bootstrap_ci, event_recall, false_alarms_per_hour,
                        frame_scores, metrics)  # fmt: skip
from vigie.explain import CueReference
from vigie.fusion import NormalReference, fuse
from vigie.scorer import Scorer
from vigie.models.classifier import predict, train_classifier
from vigie.models.normality import load_model, score_frames
from vigie.models.supervised import feature_matrix, feature_names, importances, make_head
from vigie.windows import WIN, build_windows, window_frame_labels

TARGET_FA_PER_HOUR = 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normality", default="runs/normality_retails_fit/model.pt")
    ap.add_argument("--out", default="runs/heads")
    ap.add_argument("--normal-windows", type=int, default=30_000)
    ap.add_argument("--clf-epochs", type=int, default=12)
    ap.add_argument("--test", default="retails_real")
    args = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    norm = load_model(args.normality)
    log = lambda m: print(m, flush=True)  # noqa: E731

    # ------------------------------------------------------------------ data
    holdout = get_split("retails_holdout")
    # three disjoint thirds of the unseen normal footage (by video)
    neg_videos, cal_videos, fa_videos = holdout[0::3], holdout[1::3], holdout[2::3]
    staged = get_split("retails_staged")
    test = get_split(args.test)

    ws_st = build_windows(staged, min_len=8)
    ws_ref = build_windows(neg_videos, max_windows=args.normal_windows, seed=3)
    ws_cal = build_windows(cal_videos, max_windows=args.normal_windows, seed=4)
    ws_fa = build_windows(fa_videos)
    ws_te = build_windows(test, min_len=8)
    log(f"windows: staged {len(ws_st)}, normal-neg {len(ws_ref)}, normal-cal {len(ws_cal)}, "
        f"fa {len(ws_fa)} ({sum(v.n_frames for v in fa_videos) / 15 / 3600:.1f} h), test {len(ws_te)}")

    pf = {k: score_frames(norm, w.pose) for k, w in
          {"st": ws_st, "ref": ws_ref, "cal": ws_cal, "fa": ws_fa, "te": ws_te}.items()}  # fmt: skip

    # training set for the heads: staged windows + normal reference windows
    X = np.concatenate([feature_matrix(ws_st.cues, pf["st"]), feature_matrix(ws_ref.cues, pf["ref"])])
    y_soft = np.concatenate([ws_st.label, np.zeros(len(ws_ref), np.float32)])
    y = (y_soft >= 0.5).astype(int)
    groups = np.concatenate([ws_st.video, 10_000 + ws_ref.video])
    names = feature_names()

    # ------------------------------------------------------------------ head 1: boosted trees
    cv_pred = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_head().fit(X[tr], y[tr])
        cv_pred[va] = m.predict_proba(X[va])[:, 1]
    st_mask = np.arange(len(y)) < len(ws_st)
    log(f"GBM 5-fold CV AUC (grouped by video): all {roc_auc_score(y, cv_pred):.3f}, "
        f"staged-only {roc_auc_score(y[st_mask], cv_pred[st_mask]):.3f}")  # fmt: skip
    gbm = make_head().fit(X, y)
    imp = importances(gbm, X[st_mask], y[st_mask], names)
    log("top features (permutation importance on staged): " +
        ", ".join(f"{n} {v:.3f}" for n, v in imp[:8]))  # fmt: skip

    # ------------------------------------------------------------------ head 2: fine-tuned net
    pose_tr = np.concatenate([ws_st.pose, ws_ref.pose])
    fl_tr = np.concatenate([window_frame_labels(staged, ws_st), np.zeros((len(ws_ref), WIN), np.uint8)])
    clf = train_classifier(norm, pose_tr, y_soft, fl_tr, epochs=args.clf_epochs, log=log)
    torch.save(clf.state_dict(), out / "classifier.pt")

    # ------------------------------------------------------------------ scoring
    def scores(ws, pfr):
        g = gbm.predict_proba(feature_matrix(ws.cues, pfr))[:, 1]
        cw, cf = predict(clf, ws.pose)
        return {"normality": pfr, "gbm": np.repeat(g[:, None], WIN, 1), "finetuned": cf * 0.5 + cw[:, None] * 0.5}

    s_cal, s_fa, s_te = scores(ws_cal, pf["cal"]), scores(ws_fa, pf["fa"]), scores(ws_te, pf["te"])
    refs = {k: NormalReference(v) for k, v in s_cal.items()}
    combos = {
        "normality": ["normality"], "gbm": ["gbm"], "finetuned": ["finetuned"],
        "gbm+finetuned": ["gbm", "finetuned"], "all three": ["normality", "gbm", "finetuned"],
    }  # fmt: skip

    def fused(s, keys):
        return fuse([refs[k].logit(s[k]) for k in keys])

    results = {}
    for name, keys in combos.items():
        te = fused(s_te, keys)
        fa = fused(s_fa, keys)
        row = {}
        for sigma in (0, 5):
            sc = frame_scores(test, ws_te, te.mean(1), per_frame=te, sigma=sigma)
            m = metrics(test, sc)
            ci = bootstrap_ci(test, sc, n=500)
            row[f"sigma{sigma}"] = {**m.__dict__, "ci": ci}
        # operating point: the lowest threshold that keeps false alarms under the budget
        best = None
        for thr in np.quantile(fused(s_cal, keys), [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998]):
            pol = AlertPolicy(float(thr))
            rate, n_al, hours = false_alarms_per_hour(fa_videos, ws_fa, fa, pol)
            if rate <= TARGET_FA_PER_HOUR:
                rec, n_ev, delays = event_recall(test, ws_te, te, pol)
                best = {"threshold": float(thr), "fa_per_hour": rate, "alerts": n_al, "hours": hours,
                        "event_recall": rec, "events": n_ev,
                        "median_delay_s": float(np.median(delays)) if delays else None}  # fmt: skip
                break
        row["operating_point"] = best
        if name == "all three":  # full trade-off curve for the report
            curve = []
            for q in [0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999]:
                pol = AlertPolicy(float(np.quantile(fused(s_cal, keys), q)))
                rate, _, _ = false_alarms_per_hour(fa_videos, ws_fa, fa, pol)
                rec, _, _ = event_recall(test, ws_te, te, pol)
                curve.append({"quantile": q, "fa_per_hour": rate, "event_recall": rec})
            row["tradeoff"] = curve
        results[name] = row
        m0 = row["sigma0"]
        op = (f"recall {best['event_recall']:.2f} @ {best['fa_per_hour']:.2f} FA/h"
              if best else "no threshold met the FA budget")
        log(f"{name:14s} AUC-ROC {100 * m0['auc_roc']:5.2f} "
            f"[{100 * m0['ci']['auc_roc'][0]:.1f}-{100 * m0['ci']['auc_roc'][1]:.1f}]  "
            f"AUC-PR {100 * m0['auc_pr']:5.2f}  EER {m0['eer']:.3f} | sigma5 AUC-ROC "
            f"{100 * row['sigma5']['auc_roc']:5.2f} | {op}")  # fmt: skip

    results["features"] = imp
    (out / f"results_{args.test}.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez(out / f"scores_{args.test}.npz", all=fused(s_te, combos["all three"]))

    # the shipped system is the three-way fusion, fixed before looking at test results;
    # its threshold comes from normal footage only (false-alarm budget), never from the test set
    op = results["all three"]["operating_point"]
    scorer = Scorer(norm, clf, gbm, refs, CueReference(ws_cal.frame_cues),
                    keys=combos["all three"], threshold=op["threshold"] if op else 0.0,
                    meta={"fa_budget_per_hour": TARGET_FA_PER_HOUR, "trained_on": "RetailS"})  # fmt: skip
    scorer.save(out / "bundle")
    log(f"bundle saved to {out / 'bundle'}")


if __name__ == "__main__":
    main()
