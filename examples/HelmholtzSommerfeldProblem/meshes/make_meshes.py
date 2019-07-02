#!/usr/bin/python
from subprocess import call
import math
import os

cmd = "-2"  # "-2" or "-3"
name = "circle_in_square"  # Should be name of .geo file AND name of file
file_name = name + '/' + name + '.geo'

init_char_max = 0.5
num_refine = 0
const = 2.0


clmax = [init_char_max * pow(const, -n) for n in range(num_refine+1)]
clmin = [mx / 2 for mx in clmax]

for mx, mn in zip(clmax, clmin):
    out_name = name + '/'
    out_name += "max:%f" % (mx)
    out_name = out_name.replace('.', '%')
    out_name += '.msh'

    # if not already instantiated
    if not os.path.isfile(out_name):

        call(["gmsh", cmd,
              "-clmax", str(mx), "-clmin", str(mn),
              file_name, "-o", out_name])
