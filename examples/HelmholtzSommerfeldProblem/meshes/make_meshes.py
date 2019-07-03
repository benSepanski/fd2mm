#!/usr/bin/python
import math
import os

from os.path import abspath, dirname, join
from subprocess import call

cmd = "-3"  # "-2" or "-3"
name = "ball_in_cube"  # Should be name of .geo file AND name of folder

init_char_max = 0.5
num_refine = 0
const = 2.0

cwd = abspath(dirname(__file__))
folder_name = join(cwd, name)
in_name = join(folder_name, name + '.geo')

clmax = [init_char_max * pow(const, -n) for n in range(num_refine+1)]
clmin = [mx / 2 for mx in clmax]

for mx, mn in zip(clmax, clmin):
    out_name = "max:%f" % (mx)
    out_name = out_name.replace('.', '%')
    out_name += '.msh'
    out_name = join(folder_name, out_name)

    # if not already instantiated
    if not os.path.isfile(out_name):

        call(["gmsh", cmd,
              "-clmax", str(mx), "-clmin", str(mn),
              in_name, "-o", out_name])
