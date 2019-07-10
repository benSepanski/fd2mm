.. role:: bash(code)
    :language: bash

firedrake_to_pytential:
=======================

mesh conversion from firedrake to pytential 
-------------------------------------------

WARNING: Still in development, interface may not be final

Installation:

* First install `firedrake <https://firedrakeproject.org/download.html>`_. If you want to use things like a `HelmhotlzKernel`, you'll want to install the `complex version <https://github.com/firedrakeproject/firedrake/projects/4>`_. Note this is still in development.
* Inside the firedrake virtual environment, install pytential.
  To do this, make sure you have pyopencl and pybind11. Then, install `pytential <https://documen.tician.de/pytential/misc.html#installing-pytential>`_ starting from step 8.
* If you don't intend to be developing, run::

    pip install git+https://github.com/benSepanski/firedrake_to_pytential

* For developers, run::

    git clone git+https://github.com/benSepanski/firedrake_to_pytential
    pip install -e <path-to-cloned-directory>

* Run::
    make meshes
This may take some time.

* Before running programs using pytential, you may want to set :bash:`PYOPENCL_CTX=0`
* To run any tests, type::
    make test


Resources:

* `Firedrake Install Instructions <https://firedrakeproject.org/download.html>`_.
* `Complex Firedrake install instructions  `complex version <https://github.com/firedrakeproject/firedrake/projects/4>`_.
* `pytential install instructions <https://documen.tician.de/pytential/misc.html#installing-pytential>`_.
* `Source code <https://github.com/benSepanski/firedrake_to_pytential>`_ on github

Note:

* Setup files edited from Andreas' Klockner's `meshmode library <https://github.com/inducer/meshmode>`_
* Makefile and requirements formated based on firedrake
* To get pyopencl, for me it sufficed to install the following packages (Ubuntu 18.04):

    * https://packages.ubuntu.com/bionic/pocl-opencl-icd
    * https://packages.ubuntu.com/bionic/ocl-icd-libopencl1
    * https://packages.ubuntu.com/bionic/opencl-headers
 
