import numpy as np
from utilities import load 
from matlab_erm.visualizeBlockCore import *

# Script
dataset = load("data/python_erm/BlockSimData.pkl")
# First convert all the source labels to strings
for data in dataset:
    data['source'] = "".join([chr(code) for code in data['source'][0,:]])
# Split apart the data for the nominal case, the erm case, and the worst-case
nominal = [data for data in dataset if data['source'] == "nominal"]
erm = [data for data in dataset if data['source'] == "ERM"]
worstcase = [data for data in dataset if data['source'] == "min"]
print(f"There are {len(nominal)} nominal cases, {len(erm)} ERM cases, and {len(worstcase)} worst-case scenario cases")
# Visualize all the nominal cases
datacolors = [colors['blue'], colors['red'], colors['green']]
terraincolors = np.array([0.9, 0.6, 0.3, 0.1])
for n in range(len(nominal)):
    trajdata = [nominal[n], erm[n], worstcase[n]]
    terraincolor = np.array([terraincolors[n], terraincolors[n], terraincolors[n], 1.])
    print(f"Visualizing for friction = {nominal[n]['friction']}")
    print(f"Nominal is blue. \nERM is red\nWorst-case is green")
    visualize_multiblock(trajdata, datacolors, terraincolor)



