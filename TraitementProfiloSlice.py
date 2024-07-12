# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:46:00 2023

@author: Pauline LEFEBVRE
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
taille_police = 15
path="figures/"

fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
#ax.set_xlim(400,1600)
#ax.set_ylim(0,500)
ax.axis('on')
ax.set_xlabel("x (mm)", fontsize=taille_police)
ax.set_ylabel("z (µm)", fontsize=taille_police)

columns=['X', 'Y','Slice']

slice_data = pd.read_csv(path+"PMMA-I.xyz", header=1, names=columns)
x = slice_data['X']
z = slice_data['Y']
j=0
xbis=[]
zbis=[]
for i in range(0,len(z)-1):
    if z[i]!='                   ':
        zbis.append(float(z[i]))
        xbis.append(float(x[i]))

plt.plot(xbis, zbis,'-b', markersize=2, label='-1 mesuré')

plt.savefig(path + 'PMMA-I2.png',dpi=300)