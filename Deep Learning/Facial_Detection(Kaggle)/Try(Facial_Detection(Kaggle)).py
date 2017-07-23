from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np
import sys
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

print_tensors_in_checkpoint_file(tensor_name=None,file_name='C:\\Users\\alien\Desktop\Deep_Learning_Data\Model\Facial Detection\CNN(50,5x3x3x3,36,512x512,50000)',all_tensors=True)

def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    pgmf.readline()
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

