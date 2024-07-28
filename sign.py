import numpy as np
import sys
import math
import ROOT
import pandas as pd

#per dr bhabha e vertice assunto piatto

def poissonian_chi2_high_b(s, b):
  n = s+b
  return 2*(n*np.log(n/b) + b - n)

def get_sign_from_s_b(s, b):
  return np.sqrt(poissonian_chi2_high_b(s, b))
