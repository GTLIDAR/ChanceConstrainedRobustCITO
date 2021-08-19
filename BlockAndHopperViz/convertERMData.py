import numpy as np
import h5py
from utilities import save

def convert_mat_to_pkl(filename):
    """
        Convert matlab .mat file to .pkl file

        Currently converts structure arrays of dimension 1 or 2 to lists of python dictionaries.
    """
    data = {}
    with h5py.File(filename, "r") as file:
        datakey = list(file.keys())[1]
        fieldkeys = list(file[datakey].keys())
        nfields = len(file[datakey][fieldkeys[0]])
        if nfields == 1:
            data = {}
            for key in fieldkeys:
                data[key] = np.array(file[datakey][key]).transpose()   
            data = [data]
        else:
            data = [0]*nfields
            for n in range(nfields):
                data[n] = {}
                for key in fieldkeys:
                    ref = file[datakey][key][n].item()
                    data[n][key] = np.array(file[ref]).transpose()  
    return data

if __name__ == "__main__":
    # Setup each of the file parts
    filepath = 'data/matlab_erm/'
    files = ['PushBlockNominal', 'PushBlockERM', 'HopperNominal','HopperERM','CartNominal','CartERM','BlockSimData']
    source_ext = 'ForPython.mat'
    targetpath = 'data/python_erm/'
    target_ext = '.pkl'
    # Read and convert each of the files
    for file in files:
        data = convert_mat_to_pkl(filepath+file+source_ext)
        save(targetpath+file+target_ext, data)


