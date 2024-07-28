import ROOT
import uproot
import numpy as np

f = ROOT.TFile("sig_pretrain.root")
h = f.Get("sig")

def inverse_weight(mass):
  return 1/h.GetBinContent(h.FindBin(mass))

_inverse_weight = np.vectorize(inverse_weight)

ff = uproot.open("smeared_sig_5mgen_with_score.root")

mass = ff["tree"]["mass"].array(library="np")
weight = ff["tree"]["weight"].array(library="np")

weight *= _inverse_weight(mass)

fff = uproot.recreate("/tmp/prova.root")
fff["tree"] = {"mass": mass, "weight": weight}
fff.close()
