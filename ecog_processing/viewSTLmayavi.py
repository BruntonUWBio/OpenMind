"""
To read a STL file and plot in mayavi
First created by Junwei Huang @ Toronto, Feb 26, 2013

Modified to plot landmarks.

Usage: python viewSTLmayavi.py both_ascii.stl trodes.txt

"""


from numpy import *
from mayavi import mlab
import pandas as pd
import sys

from scipy.io import loadmat


def get_points(mat_file):
    mat = loadmat(mat_file)
    pts = {}
    pts["x"] = [x[0] for x in mat['AllTrodes']]
    pts["y"] = [x[1] for x in mat['AllTrodes']]
    pts["z"] = [x[2] for x in mat['AllTrodes']]
    return pts

def get_mayavi_fig(STLfile, Matfile):
    f = open(STLfile, 'r')

    x = []
    y = []
    z = []
    limit = 1000000
    for num, line in enumerate(f):
        # if num > limit:  # line_number starts at 0.
        # break
        strarray = line.split()
        if strarray[0] == 'vertex':
            x = append(x, double(strarray[1]))
            y = append(y, double(strarray[2]))
            z = append(z, double(strarray[3]))

    triangles = [(i, i + 1, i + 2) for i in range(0, len(x), 3)]

    mlab.triangular_mesh(x, y, z, triangles)
    # pts = pd.read_csv(sys.argv[2])
    pts = get_points(Matfile)

    mlab.points3d(pts["x"], pts["y"], pts["z"], scale_factor=5)
    return mlab.gcf()

if __name__ == '__main__':
    STLfile = sys.argv[1]
    Matfile = sys.argv[2]
    # pts = get_points(sys.argv[2])
    get_mayavi_fig(STLfile, Matfile)
    mlab.show()
