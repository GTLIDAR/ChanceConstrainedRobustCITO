"""
Luke Drnach
September 8, 2021
"""
import os, shutil
import utilities as utils
from robusthoppingopt import RobustOptimizationOptions, copy_config_to_data

def correct():
    dir = os.path.join("examples","hopper","robust_erm")
    file1 = os.path.join(dir, "linear_NCC_sigma_3e-01_nochance","trajoptresults.pkl")
    file2  = os.path.join(dir, 'linear_NCC_sigma_3e-02_nochance','trajoptresults.pkl')
    data1 = utils.load(file1)
    data2 = utils.load(file2)
    data1['config'] = data2['config']
    data1['config'].sigma = 3e-01
    utils.save(file1, data1)

def split_config_from_data(directory, filename = 'trajoptresults.pkl'):
    """Separate configuration data from trajectory optimization results and save separately"""
    if not os.path.isdir(directory):
        print(f"{directory} not found")
    for filepath in utils.find_filepath_recursive(directory, filename):
        fullpath = os.path.join(filepath, filename)
        print(f"Organizing file: {fullpath}")
        data = utils.load(fullpath)
        # Separate out the configuration from the data
        if 'config' in data.keys():
            config = data.pop('config')
            # Add back in run information
            data = copy_config_to_data(config, data)
            # Resave the data and configuration separately
            utils.save(fullpath, data)
            utils.save(os.path.join(filepath, 'config.pkl'), config)
        elif 'sigma' not in data.keys():
            configpath = os.path.join(filepath, 'config.pkl')
            config = utils.load(configpath)
            data = copy_config_to_data(config, data)
            utils.save(fullpath, data)

def organize_solutions_by_success(directory, filename='trajoptresults.pkl'):
    olddirs = []
    newdirs = []
    if not os.path.isdir(directory):
        print(f"{directory} does not exist")
    # Create the output directories
    successdir = os.path.join(directory, 'success')
    faildir = os.path.join(directory, 'fail')
    if not os.path.isdir(successdir):
        os.makedirs(successdir)
    if not os.path.isdir(faildir):
        os.makedirs(faildir)
    # Check each file
    for filepath in utils.find_filepath_recursive(directory, filename):
        if successdir in filepath or faildir in filepath:
            print(f"{filepath} has already been sorted. Skipping.")
        else:
            fullpath = os.path.join(filepath, filename)
            print(f"Organizing file: {fullpath}")
            data = utils.load(fullpath)
            if data['success']:
                newdir = os.path.join(successdir, filepath.replace(directory + os.sep,""))
            else:
                newdir = os.path.join(faildir, filepath.replace(directory + os.sep, ""))
            olddirs.append(filepath)
            newdirs.append(newdir)
    # Move the directories
    for olddir, newdir in zip(olddirs, newdirs):
        print(f"Moving directory {olddir}")
        shutil.move(olddir, newdir)
    return successdir

if __name__ == "__main__":
    dir  = os.path.join("examples","hopper","robust_nonlinear","cc_erm_1e5_mod")
    split_config_from_data(dir)
    organize_solutions_by_success(dir)