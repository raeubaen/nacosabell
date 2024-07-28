import ROOT
import numpy as np
import sign
import pandas as pd

sig_file = ROOT.TFile("../smeared_sig_5mgen_with_score.root")
sig_tree = sig_file.Get("tree")
sig_tree.SetName("sig")
bkg_file = ROOT.TFile("../smeared_bkg_noresample_42179e3_with_score.root")
bkg_tree = bkg_file.Get("tree")
bkg_tree.SetName("bkg")

thrs = np.linspace(0, 1, 100)[:-1]
zs = []
sigs = []
bkgs = []

bkg_mass = ROOT.TH1F("bkg_mass", "bkg_mass", 40, 0.212-0.04, 0.212+0.04)
sig_mass = ROOT.TH1F("sig_mass", "sig_mass", 40, 0.212-0.04, 0.212+0.04)

for thr in thrs:
  sig_mass.Reset()
  bkg_mass.Reset()
  sig_tree.Draw("mass>>sig_mass", f"weight * 53 * 500 / 5e6 * (score > {thr})", "goff")
  bkg_tree.Draw("mass>>bkg_mass", f"weight * 186e3 * 500 / 42179e3 * (score > {thr})", "goff")
  sigs.append(sig_mass.Integral())
  bkgs.append(bkg_mass.Integral())
  print(thr, sig_mass.Integral(), bkg_mass.Integral())
  z = sign.get_sign_from_s_b(sig_mass.Integral(), bkg_mass.Integral())
  print(z, sig_mass.Integral()/np.sqrt(bkg_mass.Integral()), sig_mass.Integral()/np.sqrt( bkg_mass.Integral() + (bkg_mass.Integral()*0.01)**2 ) )
  zs.append(z)

df = pd.DataFrame({"z": zs, "thr": thrs, "s": sigs, "b": bkgs})
df.to_csv("../opt.csv", index=None)
g = ROOT.TGraph(len(zs), thrs.astype(float), np.asarray(zs).astype(float))
g.Draw()
g.SaveAs("../g_opt.root")
input()
