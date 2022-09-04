#!/bin/bash
# Script to compile and execute a C API program
python3 setup.py build_ext --inplace
python3 setup.py install
