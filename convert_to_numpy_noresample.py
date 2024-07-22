import ROOT
import numpy as np

c = ROOT.TChain("tree")
c.Add("/eos/user/r/rgargiul/LbL_belle2/LbL_noresample_smeared.root")
df = ROOT.RDataFrame(c)
df_n = df.AsNumpy()
print(df_n)
matrix = np.vstack(list(df_n.values())).T
np.save(f"LbL_42179e3_smeared.npy", matrix)
