import ROOT

sig_file = ROOT.TFile("../smeared_sig_5mgen_with_score.root")
sig_tree = sig_file.Get("tree")
sig_hist = ROOT.TH1F("sig", "sig", 40, 0, 0.4)
sig_tree.Draw("mass>>sig", "55*500/5e6 * weight * (score > 0.6)")
sig_hist.SaveAs("../sig_hist.root")
sig_file.Close()
bkg_file = ROOT.TFile("../smeared_bkg_noresample_42179e3_with_score.root")
bkg_tree = bkg_file.Get("tree")
bkg_hist = ROOT.TH1F("bkg", "bkg", 40, 0, 0.4)
bkg_tree.Draw("mass>>bkg", "500 * 186e3 / 42179e3 * weight * (score > 0.6)")
bkg_hist.SaveAs("../bkg_hist.root")
