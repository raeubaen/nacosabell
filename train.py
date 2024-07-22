import time
import uproot

import cupy as cp
from sklearn.model_selection import train_test_split
import numpy as np

import xgboost as xgb

from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train', action='store_true')
args = parser.parse_args()


bkg = np.load("LbL_42179e3_smeared.npy")[:, 1:-1]

bkg = np.delete(bkg, 5, 1)

bkg = np.delete(bkg, 2, 1)

sig = np.load("TM_smeared.npy")[:, :-1]

sig = np.delete(sig, 5, 1)

sig = np.delete(sig, 2, 1)

bkg_label = np.zeros(len(bkg))
sig_label = np.ones(len(sig))

X = np.concatenate((bkg, sig))
y = np.concatenate((bkg_label, sig_label))
print(X.shape)
print(y.shape)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.9, train_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, train_size=0.9, random_state=42
)

print(y_test.shape)
print(y_test)

# Specify sufficient boosting iterations to reach a minimum
num_round = 2500

clf = xgb.XGBClassifier(device="cuda", n_estimators=num_round)

if args.train:
    X_train = cp.asarray(X_train)
    y_train = cp.array(y_train)
    X_val = cp.array(X_val)
    y_val = cp.array(y_val)
    X_test = cp.array(X_test)
    y_test = cp.array(y_test)

    start = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])

    clf.save_model("model.json")
    preds = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(cp.asnumpy(y_test), preds)
else:
    clf.load_model("model.json")
    preds = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, preds)

sig_preds = clf.predict_proba(sig)[:, 1]
print("sig", sig.shape)
passing_signals = sig[sig_preds > 0.27]
print("pass sig", passing_signals.shape)
bkg_preds = clf.predict_proba(bkg)[:, 1]
print("bkg", bkg.shape)
passing_bkgs = bkg[bkg_preds > 0.27]
print("pass bkg", passing_bkgs.shape)

sig = np.load("TM_smeared.npy")

sig_with_preds = np.hstack((sig, sig_preds.reshape(len(sig_preds), 1)))

df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thr": thresholds})
df.to_csv("roc.csv")

branches = ['lead_en', 'lead_eta', 'lead_phi', 'lead_pt', 'lead_theta', 'mass', 'pt', 'sublead_en', 'sublead_eta', 'sublead_phi', 'sublead_pt', 'sublead_theta', "weight", "score"]

sig_tree_dict = {branches[i]: sig_with_preds[:, i] for i in range(len(branches))}

sig_file = uproot.recreate("smeared_sig_5mgen_with_score_try2.root")

sig_file["tree"] = sig_tree_dict

sig_file.close()


bkg = np.load("LbL_42179e3_smeared.npy")[:, :-1]

branches = ['weight', 'lead_en', 'lead_eta', 'lead_phi', 'lead_pt', 'lead_theta', 'mass', 'pt', 'sublead_en', 'sublead_eta', 'sublead_phi', 'sublead_pt', 'sublead_theta', "score"]

bkg_with_preds = np.hstack((bkg, bkg_preds.reshape(len(bkg_preds), 1)))

bkg_tree_dict = {branches[i]: bkg_with_preds[:, i] for i in range(len(branches))}

bkg_file = uproot.recreate("smeared_bkg_noresample_42179e3_with_score.root")

bkg_file["tree"] = bkg_tree_dict

bkg_file.close()
