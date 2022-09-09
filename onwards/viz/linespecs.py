MYBLUE = [0.25, 0.25, 0.75]
MYRED  = [0.8, 0.1, 0.3]

REF = {'linewidth':0.7, 'color':'k', 'label': 'BigFlow'}
MOD = {'linewidth':1, 'color':MYBLUE, 'label': 'Flow Model'}
WT = {'linewidth':2.5, 'color':'k'}
REFscat = {'color':'k', 'label': 'BigFlow'}
MODscat = {'color':MYBLUE, 'label': 'Flow Model'}

import matplotlib.pyplot as plt
CMAP =  plt.get_cmap('viridis')
CMAP_VEL =  plt.get_cmap('seismic')