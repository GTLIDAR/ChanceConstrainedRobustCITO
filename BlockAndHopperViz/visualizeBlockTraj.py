import numpy as np
from utilities import load  
from visualizeBlockCore import colors, visualize_multiblock
import pickle
import matplotlib.pyplot as plt

# ermdata = load("data/python_erm/PushBlockERM.pkl")
# nominal = load("data/python_erm/PushBlockNominal.pkl")
nominal = np.loadtxt('data/slidingblock/warm_start/x.txt')
ermdata = np.loadtxt('data/slidingblock/erm/horizontal_position.txt')
t = np.loadtxt('data/slidingblock/erm/t.txt')

horizontal_position = ermdata[1,:]
vertical_position = np.ones(horizontal_position.shape)*0.5
vertical_velocity = np.zeros(horizontal_position.shape)
horizontal_velocity = np.zeros(horizontal_position.shape)
# horizontal_velocity[0] = 0
# horizontal_velocity[1:] = (horizontal_position[1:] - horizontal_position[:-1])/0.01
# fig1, axs1 = plt.subplots(1,1)
# axs1.plot(t, nominal[1])
# plt.show()

ermdata = np.array([horizontal_position, vertical_position, horizontal_velocity, vertical_velocity])

print(ermdata.shape)
# dataset = [nominal[0], ermdata[0], ermdata[1], ermdata[2]]
dataset = [nominal,ermdata]
# colornames = ['blue','purple','yellow','red']
colornames = ['blue', 'purple']
datacolors = [colors[name] for name in colornames]
terraincolor = colors['medgrey']
visualize_multiblock(dataset, datacolors, terraincolor, key='xtraj')

print(f"Visualized trajectory data")
print(f"Nominal shown in {colornames[0]}")
for n in range(1,4):
    print(f"ERM with sigma = {dataset[n]['sigma']} in {colornames[n]}")
