import os, sys, pathlib
file = pathlib.Path(__file__)
parent = file.parents[2]
sys.path.append(str(parent))