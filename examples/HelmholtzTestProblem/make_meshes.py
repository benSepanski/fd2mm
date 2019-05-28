#!/usr/bin/python
from subprocess import call
import math
import os

mesh_file = "domain"
ext = ".geo"
out_folder = "msh_files/"
init_char_max = 0.0005
num_refine = 0
const = 2.0


clmax = [pow(const, -n) for n in range(num_refine+1)]
clmin = [init_char_max * m / 2 for m in clmax]

for mx, mn in zip(clmax, clmin):
    out_name = out_folder
    out_name += mesh_file + "--max:%f,min%f" % (mx, mn)
    out_name = out_name.replace('.', '%')
    out_name += '.msh'

    # if not already instantiated
    if not os.path.isfile(out_name):

        call(["gmsh", "-2",
              "-clmax", str(mx), "-clmin", str(mn),
              mesh_file + ext, "-o", out_name])
