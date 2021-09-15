from robusthoppingopt import RobustOptimizationOptions
from utilities import load
import os


def main(directory):
    file = os.path.join(directory, 'config.pkl')
    config = load(file)
    print(f"Configuration {file}")
    attrs = vars(config)
    for key in attrs.keys():
        if isinstance(attrs[key], dict):
            for dkey in attrs[key].keys():
                print(f"{key}[{dkey}]: {attrs[key][dkey]}")
        else:
            print(f"{key}: {attrs[key]} ")

if __name__ == '__main__':
    dir = os.path.join("examples","hopper","robust_erm_hotfix_1e6_linear","success","erm","linear_NCC_sigma_1e-01_nochance")
    main(dir)