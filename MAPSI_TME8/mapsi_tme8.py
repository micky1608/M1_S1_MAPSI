#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:14:15 2018

@author: 3873418
"""

import numpy as np
import pickle as pkl
from markov_tools import *


with open('data_tme8.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1')

Xgenes  = data.get("genes") #Les genes, une array de arrays

Genome = data.get("genome") #le premier million de bp de Coli

Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"]

#####################################################################################

a = 1/200

b_proba = []

for gene in Xgenes:
    b_proba.append(3/len(gene))

b = np.mean(b_proba)

Pi = np.array([1, 0, 0, 0])
A =  np.array([[1-a, a  , 0, 0],
              [0  , 0  , 1, 0],
              [0  , 0  , 0, 1],
              [b  , 1-b, 0, 0 ]])

print("Pi : \n",Pi)
print("A : \n",A)


intergene_distribution = np.zeros(4)

for bp in Genome:
    intergene_distribution[bp] += 1

intergene_distribution /= 1000000

print("\nIntergene distribution : :,", intergene_distribution)


gene_distribution = np.zeros((3,4))
total_bp = np.zeros(3)

for gene in Xgenes:
    for i in range(3,len(gene)-3):
        pos = i%3
        gene_distribution[pos,gene[i]] += 1
        total_bp[pos] += 1

for i in range(3):
    gene_distribution[i,:] /= total_bp[i]

print("Gene distribution : \n",gene_distribution)

