import ROOT
import numpy as np
# import tqdm
import time
import argparse
import argparse
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
import argparse


def tree_var(tree, name, shape, npvartype, rootvartype):
  dtype = npvartype
  var = np.zeros(shape, dtype=dtype)
  shape_str = "".join(["[%i]"%i for i in shape])
  tree.Branch(name, var, "%s%s/%s"%(name,shape_str,rootvartype))
  return var


parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.add_argument("boost")

args = parser.parse_args()


outfile = args.outfile
infile = args.infile
boost = int(args.boost)

outf = ROOT.TFile(outfile, "RECREATE")
outf.cd()
tree = ROOT.TTree("tree", "tree")

tree.SetAutoSave(1000)

f = ROOT.TFile(infile)
intree = f.Get("mytree")

maxevents = min(int(1e6), intree.GetEntries())


#new tree branches definition using the function tree_var
#def tree_var(tree, name, shape, npvartype, rootvartype):
tree_weight = tree_var(tree, "weight", (1,),  np.float32, "F")
tree_mass = tree_var(tree, "mass", (1,),  np.float32, "F")
tree_lead_pt = tree_var(tree, "lead_pt", (1,),  np.float32, "F")
tree_lead_phi = tree_var(tree, "lead_phi", (1,),  np.float32, "F")
tree_lead_eta = tree_var(tree, "lead_eta", (1,),  np.float32, "F")
tree_lead_theta = tree_var(tree, "lead_theta", (1,),  np.float32, "F")
tree_lead_en = tree_var(tree, "lead_en", (1,),  np.float32, "F")
tree_pt = tree_var(tree, "pt", (1,),  np.float32, "F")
tree_sublead_pt = tree_var(tree, "sublead_pt", (1,),  np.float32, "F")
tree_sublead_en = tree_var(tree, "sublead_en", (1,),  np.float32, "F")
tree_sublead_phi = tree_var(tree, "sublead_phi", (1,),  np.float32, "F")
tree_sublead_eta = tree_var(tree, "sublead_eta", (1,),  np.float32, "F")
tree_sublead_theta = tree_var(tree, "sublead_theta", (1,),  np.float32, "F")


for ev in range(maxevents):

    intree.GetEntry(ev)

    if ev%1000==0: print(f" Event {ev} out of {maxevents}")

    if intree.numParticles == 4:
        px1, px2, py1, py2, pz1, pz2 = intree.px[2], intree.px[3], intree.py[2], intree.py[3], intree.pz[2], intree.pz[3]
    elif intree.numParticles == 5:
        px1, px2, py1, py2, pz1, pz2 = intree.px[3], intree.px[4], intree.py[3], intree.py[4], intree.pz[3], intree.pz[4]
    else:
        print("Problem: numParticles != 4 and != 5")

    tree_weight[0] = intree.eventWeight

    ph1, ph2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
    ph1.SetXYZM(px1, py1, pz1, 0)
    ph2.SetXYZM(px2, py2, pz2, 0)

    if boost:
      ph1.Boost(0, 0, 0.28)
      ph2.Boost(0, 0, 0.28)

    pt1 = ph1.Pt()
    pt2 = ph2.Pt()

    if pt1 > pt2:
      lead = ph1
      sublead = ph2
    else:
      lead = ph2
      sublead = ph1

    tree_lead_pt[0] = lead.Pt()
    tree_lead_theta[0] = lead.Theta()
    tree_lead_eta[0] = lead.Eta()
    tree_lead_phi[0] = lead.Phi()
    tree_lead_en[0] = lead.E()

    tree_sublead_pt[0] = sublead.Pt()
    tree_sublead_theta[0] = sublead.Theta()
    tree_sublead_eta[0] = sublead.Eta()
    tree_sublead_phi[0] = sublead.Phi()
    tree_sublead_en[0] = sublead.E()

    tree_mass[0] = (lead+sublead).M()
    tree_pt[0] = (lead+sublead).Pt()

    tree.Fill()

outf.cd()
tree.Write()
outf.Close()
f.Close()
