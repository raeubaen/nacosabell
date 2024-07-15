import ROOT
import numpy as np

for p in ["LbL", "TM"]:
  c = ROOT.TChain("tree")
  c.Add("LbL/smeared/*.root")

  data = c.AsMatrix(["mass"])
  df = ROOT.RDataFrame(c)
  df_n = df.AsNumpy()
  matrix = np.vstack(list(df_n.values())).T
  np.save(f"{P}_smeared.npy", matrix)
