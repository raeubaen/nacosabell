import ROOT
import numpy as np
# import tqdm
import time
import argparse
import argparse
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def tree_var(tree, name, shape, npvartype, rootvartype):
  dtype = npvartype
  var = np.zeros(shape, dtype=dtype)
  shape_str = "".join(["[%i]"%i for i in shape])
  tree.Branch(name, var, "%s%s/%s"%(name,shape_str,rootvartype))
  return var


parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
args = parser.parse_args()

df = pd.read_csv("/home/ruben/Documents/tm/belle-ii/repo/calo_reso_gnn_arxiv2306.04179_tab2.csv")

outfile = args.outfile
infile = args.infile

outf = ROOT.TFile(outfile, "RECREATE")
outf.cd()
tree = ROOT.TTree("tree", "tree")

tree.SetAutoSave(1000)

f = ROOT.TFile(infile)
intree = f.Get("tree")

maxevents = min(int(1e6), intree.GetEntries())


#new tree branches definition using the function tree_var
#def tree_var(tree, name, shape, npvartype, rootvartype):
tree_mass = tree_var(tree, "mass", (1,),  np.float32, "F")
tree_lead_pt = tree_var(tree, "lead_pt", (1,),  np.float32, "F")
tree_lead_en = tree_var(tree, "lead_en", (1,),  np.float32, "F")
tree_lead_phi = tree_var(tree, "lead_phi", (1,),  np.float32, "F")
tree_lead_eta = tree_var(tree, "lead_eta", (1,),  np.float32, "F")
tree_lead_theta = tree_var(tree, "lead_theta", (1,),  np.float32, "F")
tree_pt = tree_var(tree, "pt", (1,),  np.float32, "F")
tree_sublead_en = tree_var(tree, "sublead_en", (1,),  np.float32, "F")
tree_sublead_pt = tree_var(tree, "sublead_pt", (1,),  np.float32, "F")
tree_sublead_phi = tree_var(tree, "sublead_phi", (1,),  np.float32, "F")
tree_sublead_eta = tree_var(tree, "sublead_eta", (1,),  np.float32, "F")
tree_sublead_theta = tree_var(tree, "sublead_theta", (1,),  np.float32, "F")
tree_weight = tree_var(tree, "weight", (1,),  np.float32, "F")

'''
mytree->Draw(
  "sqrt((1+rng*sqrt(pow(0.15/sqrt(E[4]), 2) + pow(0.008, 2)))*(1+rng*sqrt(pow(0.15/sqrt(E[5]), 2) + pow(0.008*rng, 2))))*sqrt(pow(E[4]+E[5], 2) - pow(px[4]+px[5], 2) - pow(py[4]+py[5>
  "eventWeight * 2.47  * 10 * (numParticles==6)"
);
'''

for ev in range(maxevents):

    intree.GetEntry(ev)

    if ev%1000==0: print(f" Event {ev} out of {maxevents}")

    tree_weight[0] = intree.weight
    lt = intree.lead_theta * 180/np.pi
    ldf1 = df[df.tmin < lt]
    ldf2 = ldf1[ldf1.tmax > lt]
    if len(ldf2) == 0: continue
    la, lb, lc = [ldf2[s].iloc[0] for s in ["a", "b", "c"]]
    tree_weight[0] *= ldf2.eff.iloc[0]

    lead_smear = (1 + ROOT.gRandom.Gaus()*np.sqrt( (la/intree.lead_en)**2 + (lb/np.sqrt(intree.lead_en))**2 + lc**2 ))
    tree_lead_en[0] = intree.lead_en * lead_smear

    tree_lead_theta[0] = intree.lead_theta + ROOT.gRandom.Gaus()*0.013
    tree_lead_phi[0] = intree.lead_phi + ROOT.gRandom.Gaus()*0.013

    tree_lead_pt[0] = tree_lead_en[0] * np.sin(tree_lead_theta[0])
    tree_lead_eta[0] = -np.log(np.tan(tree_lead_theta[0]/2))

    slt = intree.sublead_theta * 180/np.pi
    sldf1 = df[df.tmin < slt]
    sldf2 = sldf1[sldf1.tmax > slt]
    if len(sldf2) == 0: continue
    sla, slb, slc = [sldf2[s].iloc[0] for s in ["a", "b", "c"]]
    tree_weight[0] *= sldf2.eff.iloc[0]

    sublead_smear = (1 + ROOT.gRandom.Gaus()*np.sqrt( (sla/intree.sublead_en)**2 + (slb/np.sqrt(intree.sublead_en))**2 + slc**2 ))
    tree_sublead_en[0] = intree.sublead_en * sublead_smear

    tree_sublead_theta[0] = intree.sublead_theta + ROOT.gRandom.Gaus()*0.013
    tree_sublead_phi[0] = intree.sublead_phi + ROOT.gRandom.Gaus()*0.013

    tree_sublead_pt[0] = tree_sublead_en[0] * np.sin(tree_sublead_theta[0])
    tree_sublead_eta[0] = -np.log(np.tan(tree_sublead_theta[0]/2))

    if not (
      (tree_lead_theta[0] >= 17/180*np.pi) and (tree_sublead_theta[0] >= 17/180*np.pi)
        and
      (tree_lead_pt[0] > 0.1) and (tree_sublead_pt[0] > 0.1)
        and
      (abs(tree_lead_theta[0] - tree_sublead_theta[0]) > 48e-3) and (abs(tree_lead_phi[0] - tree_sublead_phi[0]) > 48e-3)
    ):
      continue

    lead, sublead = ROOT.TLorentzVector(), ROOT.TLorentzVector()
    lead.SetPtEtaPhiM(tree_lead_pt[0], tree_lead_eta[0], tree_lead_phi[0], 0)
    sublead.SetPtEtaPhiM(tree_sublead_pt[0], tree_sublead_eta[0], tree_sublead_phi[0], 0)

    tree_mass[0] = (lead+sublead).M()
    tree_pt[0] = (lead+sublead).Pt()

    tree.Fill()

outf.cd()
tree.Write()
outf.Close()
f.Close()
