#!/bin/bash
# Script to compile and execute the c api program
python3 setup.py build_ext --inplace
python3 setup.py install
cd ..
python3 src/spkmeans.py 0 spk src/project_comprehensive_test/testfiles/spk_0.txt
