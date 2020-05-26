# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:58:31 2020

@author: peter
"""

import pickle
file="./valid_id.pickle"
f=pickle.load(open(file,"rb"))
outfile=file.rstrip(".pickle")+".txt"
with open(outfile,"w") as out:
    for i in f:
        out.write(i+"\n")