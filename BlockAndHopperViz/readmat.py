import h5py
import numpy as np
from utilities import save

data = {'time': [],
        'states': [],
        'controls': [],
        'forces': [],
        'jointlimits': [],
        'slacks': [],
        }

with h5py.File("data/matlab_erm/FootedHopperData.mat","r") as f:
    for key in data.keys():
        data[key] = np.array(f[key]).transpose()

# Convert the data structure into something we could have gotten from contactimplicit.py
force = np.zeros((12, 101))
slacks = np.zeros((12, 101))
outmap = [0, 1, 2, 4, 6, 8, 10, 11]
inmap = [0, 4, 1, 2, 5, 6, 3, 7]
force[outmap,1:] = data['forces'][inmap, :-1]
force[outmap, 0] = data['forces'][inmap,-1]
slacks[outmap,1:] = data['slacks'][inmap,:]
# force[0,:] = data['forces'][0,:]
# force[1,:] = data['forces'][4,:]
# force[2,:] = data['forces'][1,:]
# force[4,:] = data['forces'][2,:]
# force[6,:] = data['forces'][5,:]
# force[8,:] = data['forces'][6,:]
# force[10,:] = data['forces'][3,:]
# force[11,:] = data['forces'][7,:]
# Adjust the joint angle data
data['states'][4,:] -= np.pi/2
# Negate the joint angles
data['states'][2:5,:] = -1*data['states'][2:5,:]
data['states'][7:,:] = -1*data['states'][7:,:]

pycito_data = {'time': np.squeeze(data['time']),
                'state': data['states'],
                'control': data['controls'],
                'force': force,
                'jointlimit': data['jointlimits'],
                'slacks': slacks}



save('data/FootedHopper/FootedHopperData.pkl', pycito_data)
