#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup, find_packages

    version_dict = {}
    init_filename = "fd2mm/version.py"
    exec(compile(open(init_filename, "r").read(), init_filename, "exec"),
            version_dict)

    setup(name="fd2mm",
          version=version_dict["VERSION_TEXT"],
          description="Convert firedrake meshes and functions to meshmode",
          long_description=open("README.rst", "rt").read(),
          author="Benjamin Sepanski",
          author_email="Ben_Sepanski@Baylor.edu",
          license="",
          url="https://github.com/benSepanski/fd2mm",
          classifiers=[
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'Natural Language :: English',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.2',
              'Programming Language :: Python :: 3.3',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Software Development :: Libraries',
              'Topic :: Utilities',
              ],

          packages=find_packages(),
          install_requires=[
              "numpy",
              "modepy",
              "gmsh_interop",
              "six",
              "pytential",
              "pytools>=2018.4",
              "pytest>=2.3",
              "loo.py>=2014.1",
              ],
          )


if __name__ == '__main__':
    main()
