import tensorflow as tf
import numpy as np
import sys
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle


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

