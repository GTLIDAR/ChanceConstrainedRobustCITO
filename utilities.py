from os import path 
from sys import exit

def FindResource(filename):
    if not path.isfile(filename):
        exit(f"{filename} not found")
    else:
        return path.abspath(filename)
    